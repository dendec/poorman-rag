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
    vector: np.ndarray
    text: Optional[str] = None
    fts_text: Optional[str] = None
    metadata: Dict[str, Any] = None

class Storage:
    def __init__(self, config: IndexingConfig, db_file: str = None, index_file: str = None):
        self.cfg = config
        self._db_file = db_file or config.db_file
        self._index_file = index_file or config.index_file
        self._ensure_dirs()
        self.conn = self._init_db()
        self.index = self._init_index()
        self.current_id = self._get_db_count()
        self._check_consistency()

    def _ensure_dirs(self):
        """Creates parent directories for all storage files if they don't exist."""
        for file_path in [self._db_file, self._index_file]:
            if file_path:
                parent = os.path.dirname(file_path)
                if parent and not os.path.exists(parent):
                    logger.info(f"Creating storage directory: {parent}")
                    os.makedirs(parent, exist_ok=True)

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS dataset (id INTEGER PRIMARY KEY, text TEXT, fts_text TEXT, metadata TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        return conn

    def _init_index(self) -> Optional[Index]:
        if not getattr(self.cfg, "enable_vector_index", True):
            return None
        idx = Index(ndim=self.cfg.vector_dim, metric=self.cfg.index_metric, dtype=self.cfg.vector_dtype)
        if os.path.exists(self._index_file):
            logger.info(f"Loading vector index from {self._index_file}...")
            idx.load(self._index_file)
        return idx

    def _get_db_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM dataset")
        return cur.fetchone()[0]
    
    def get_meta(self, key: str) -> int:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def get_meta_str(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def save_meta_str(self, key: str, value: str):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _check_consistency(self):
        db_count = self.current_id
        if not getattr(self.cfg, "enable_vector_index", True) or self.index is None:
            if db_count == 0:
                logger.debug(f"New storage initialized at {self._db_file}")
            else:
                logger.info(f"Storage ready: DB={db_count} (Vector index disabled)")
            return

        idx_count = len(self.index)
        if db_count == 0 and idx_count == 0:
            logger.debug(f"New storage initialized at {self._db_file}")
        else:
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
            if self.index is not None:
                keys = np.array([r.id for r in records], dtype=np.uint64)
                vectors = np.array([r.vector for r in records])
                self.index.add(keys, vectors)

            # 2. SQLite
            # Serialize metadata to JSON strings
            data = [(r.id, r.text, r.fts_text, json.dumps(r.metadata) if r.metadata else None) for r in records]
            with self.conn:
                self.conn.executemany("INSERT INTO dataset (id, text, fts_text, metadata) VALUES (?, ?, ?, ?)", data)
                self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('source_cursor', ?)", (new_cursor,))

            # 3. Save to disk
            if self.index is not None:
                self.index.save(self._index_file)
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
                    # CONTENT-INDEXED MODE: Stores a copy of text (Fast but large)
                    self.conn.execute("""
                        CREATE VIRTUAL TABLE dataset_fts USING fts5(
                            id UNINDEXED, text, tokenize='trigram'
                        )
                    """)
                    self.conn.execute("""
                        INSERT INTO dataset_fts(id, text) 
                        SELECT id, COALESCE(fts_text, text) FROM dataset
                    """)
                else:
                    # EXTERNAL-CONTENT MODE (e.g. "size"): References 'dataset' table (Slow but small)
                    self.conn.execute("""
                        CREATE VIRTUAL TABLE dataset_fts USING fts5(
                            text, content='dataset', content_rowid='id', tokenize='trigram'
                        )
                    """)
                    # For external content mode, we still need to rebuild
                    self.conn.execute("INSERT INTO dataset_fts(dataset_fts) VALUES('rebuild')")

                self.conn.execute("INSERT INTO dataset_fts(dataset_fts) VALUES('optimize')")
            logger.info(f"FTS5 index ready in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"FTS Build Error: {e}", exc_info=True)

    def load_hashes(self, processor: Optional[TextProcessor] = None) -> set[bytes]:
        hashes = set()
        cur = self.conn.execute("SELECT text FROM dataset")
        for (raw_text,) in cur:
            text = processor.clean(raw_text) if processor else raw_text
            if text:
                hashes.add(hashlib.md5(text.encode("utf-8")).digest())
        return hashes

    def optimize(self):
        logger.info("Optimizing SQLite database (VACUUM)...")
        start = time.time()
        self.conn.execute("VACUUM")
        logger.info(f"Optimization complete in {time.time() - start:.2f}s")

    def flush(self):
        """Lightweight close: commit WAL and release file handles. No VACUUM/compression.
        Used when evicting storage from LRU cache during indexing."""
        self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self.conn.close()
        if hasattr(self, "index"):
            del self.index
        logger.debug(f"Flushed storage: {self._db_file}")

    def close(self):
        """Full finalization: VACUUM, compress, upload. Called at end of indexing."""
        self.optimize()
        self.conn.close()
        
        # Explicitly release vector index (important for FUSE/mmap)
        if hasattr(self, "index"):
            del self.index
        
        if self.cfg.db_file.endswith(".sqlite"):
            zst_file = self._db_file + ".zst"
            logger.info(f"Compressing database: {self._db_file} -> {zst_file}")
            start = time.time()
            cctx = zstd.ZstdCompressor(level=3)
            with open(self._db_file, 'rb') as f_in:
                with open(zst_file, 'wb') as f_out:
                    cctx.copy_stream(f_in, f_out)
            logger.info(f"Compression complete in {time.time() - start:.2f}s")
            
        # 4. Automate upload to S3 if configured
        uploader = get_uploader_from_config(self.cfg)
        if uploader and self.cfg.kb_alias:
            alias = self.cfg.kb_alias
            logger.info(f"🚀 Automating upload for KB: {alias}")
            
            # Upload DB (.zst if exists, else .sqlite)
            db_local = self._db_file + ".zst" if os.path.exists(self._db_file + ".zst") else self._db_file
            db_ext = ".sqlite.zst" if db_local.endswith(".zst") else ".sqlite"
            uploader.upload_file(db_local, f"rag/index/{alias}/dataset{db_ext}")
            
            # Upload Index
            uploader.upload_file(self.cfg.index_file, f"rag/index/{alias}/vectors.usearch")