import hashlib
import sqlite3
import os
import time
import json
import logging
import zstandard as zstd
from typing import Optional, List, Dict, Any
import numpy as np
from usearch.index import Index
from dataclasses import dataclass
from core.cfg import IndexingConfig
from core.deduplicator import TextProcessor
from core.s3 import get_uploader_from_config

logger = logging.getLogger("indexer.storage")

@dataclass
class Record:
    id: int
    text: str
    vector: np.ndarray
    metadata: Dict[str, Any] = None

class Storage:
    def __init__(self, config: IndexingConfig):
        self.cfg = config
        self._ensure_dirs()
        self.conn = self._init_db()
        self.index = self._init_index()
        self.current_id = self._get_db_count()
        self._check_consistency()

    def _ensure_dirs(self):
        """Creates parent directories for all storage files if they don't exist."""
        for file_path in [self.cfg.db_file, self.cfg.index_file]:
            if file_path:
                parent = os.path.dirname(file_path)
                if parent and not os.path.exists(parent):
                    logger.info(f"Creating storage directory: {parent}")
                    os.makedirs(parent, exist_ok=True)

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.db_file)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS dataset (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER)")
        return conn

    def _init_index(self) -> Index:
        idx = Index(ndim=self.cfg.vector_dim, metric=self.cfg.index_metric, dtype=self.cfg.index_dtype)
        if os.path.exists(self.cfg.index_file):
            logger.info(f"Loading vector index from {self.cfg.index_file}...")
            idx.load(self.cfg.index_file)
        return idx

    def _get_db_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM dataset")
        return cur.fetchone()[0]
    
    def get_meta(self, key: str) -> int:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else 0

    def _check_consistency(self):
        db_count = self.current_id
        idx_count = len(self.index)
        logger.info(f"Storage consistency: DB={db_count}, Index={idx_count}")
        if db_count != idx_count:
            logger.critical(f"Data Mismatch! DB={db_count}, Index={idx_count}. Please delete index files to restart.")
            raise RuntimeError("Storage consistency check failed.")

    def save_batch(self, records: List[Record], new_cursor: int):
        if not records: 
            self._save_cursor_only(new_cursor)
            return
            
        try:
            # 1. Vector Index
            keys = np.array([r.id for r in records], dtype=np.uint64)
            vectors = np.array([r.vector for r in records])
            self.index.add(keys, vectors)

            # 2. SQLite
            # Serialize metadata to JSON strings
            data = [(r.id, r.text, json.dumps(r.metadata) if r.metadata else None) for r in records]
            with self.conn:
                self.conn.executemany("INSERT INTO dataset (id, text, metadata) VALUES (?, ?, ?)", data)
                self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('source_cursor', ?)", (new_cursor,))

            # 3. Save to disk
            self.index.save(self.cfg.index_file)
            self.current_id += len(records)
            
        except Exception as e:
            logger.error(f"Failed to save batch: {e}", exc_info=True)
            raise e

    def _save_cursor_only(self, new_cursor: int):
        with self.conn:
             self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('source_cursor', ?)", (new_cursor,))

    def build_fts_index(self):
        if self.cfg.fts_mode == "none":
            logger.info("FTS5 indexing is disabled in config.")
            return
        logger.info(f"Building FTS5 index (Mode: {self.cfg.fts_mode})...")
        start_time = time.time()
        try:
            with self.conn:
                self.conn.execute("DROP TABLE IF EXISTS dataset_fts")
                if self.cfg.fts_mode == "speed":
                    self.conn.execute("""
                        CREATE VIRTUAL TABLE dataset_fts USING fts5(
                            id UNINDEXED, text, tokenize='trigram'
                        )
                    """)
                    self.conn.execute("INSERT INTO dataset_fts(id, text) SELECT id, text FROM dataset")
                else:
                    self.conn.execute("""
                        CREATE VIRTUAL TABLE dataset_fts USING fts5(
                            text, content='dataset', content_rowid='id', tokenize='trigram'
                        )
                    """)
                    self.conn.execute("INSERT INTO dataset_fts(dataset_fts) VALUES('rebuild')")

                self.conn.execute("INSERT INTO dataset_fts(dataset_fts) VALUES('optimize')")
            logger.info(f"FTS5 index ready in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"FTS Build Error: {e}", exc_info=True)

    def load_hashes(self, processor: Optional[TextProcessor] = None) -> set[str]:
        hashes = set()
        cur = self.conn.execute("SELECT text FROM dataset")
        for (raw_text,) in cur:
            text = processor.clean(raw_text) if processor else raw_text
            if text:
                hashes.add(hashlib.md5(text.encode("utf-8")).hexdigest())
        return hashes

    def optimize(self):
        logger.info("Optimizing SQLite database (VACUUM)...")
        start = time.time()
        self.conn.execute("VACUUM")
        logger.info(f"Optimization complete in {time.time() - start:.2f}s")

    def close(self):
        self.optimize()
        self.conn.close()
        
        if self.cfg.db_file.endswith(".sqlite"):
            zst_file = self.cfg.db_file + ".zst"
            logger.info(f"Compressing database: {self.cfg.db_file} -> {zst_file}")
            start = time.time()
            cctx = zstd.ZstdCompressor(level=3)
            with open(self.cfg.db_file, 'rb') as f_in:
                with open(zst_file, 'wb') as f_out:
                    cctx.copy_stream(f_in, f_out)
            logger.info(f"Compression complete in {time.time() - start:.2f}s")
            
        # 4. Automate upload to S3 if configured
        uploader = get_uploader_from_config(self.cfg)
        if uploader and self.cfg.kb_alias:
            alias = self.cfg.kb_alias
            logger.info(f"🚀 Automating upload for KB: {alias}")
            
            # Upload DB (.zst if exists, else .sqlite)
            db_local = self.cfg.db_file + ".zst" if os.path.exists(self.cfg.db_file + ".zst") else self.cfg.db_file
            db_ext = ".sqlite.zst" if db_local.endswith(".zst") else ".sqlite"
            uploader.upload_file(db_local, f"rag/index/{alias}/content{db_ext}")
            
            # Upload Index
            uploader.upload_file(self.cfg.index_file, f"rag/index/{alias}/vectors.usearch")