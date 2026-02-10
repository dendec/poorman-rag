import hashlib
import os
import time
import json
import logging
from datetime import timedelta
from typing import Optional, List, Dict, Any
import numpy as np
import lancedb
import pyarrow as pa
from dataclasses import dataclass, asdict
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
        self.db = lancedb.connect(self.cfg.lancedb_uri)
        self.table = self._init_table()
        self.current_id = self.table.count_rows() if self.table else 0
        self._check_consistency()

    def _ensure_dirs(self):
        """Creates parent directories for storage if strictly local path."""
        # Check if URI is a local path
        if not self.cfg.lancedb_uri.startswith("s3://"):
             if not os.path.exists(self.cfg.lancedb_uri):
                 os.makedirs(self.cfg.lancedb_uri, exist_ok=True)

    def _init_table(self):
        v_type = pa.float32()
        if self.cfg.vector_dtype == "int8":
            v_type = pa.int8()
        elif self.cfg.vector_dtype == "float16":
            v_type = pa.float16()

        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(v_type, self.cfg.vector_dim)),
            pa.field("metadata", pa.string()), # Storing as JSON string for compatibility
            pa.field("cursor", pa.int64()) # To store ingestion cursor
        ])
        
        try:
            return self.db.create_table(self.cfg.table_name, schema=schema, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to open/create table: {e}")
            raise e

    def _get_db_count(self) -> int:
        return self.table.count_rows()
    
    def get_meta(self, key: str) -> int:
        # We store cursor in the separate table or just query max cursor? 
        # For simplicity, let's store metadata in a separate table "meta"
        # LanceDB doesn't support key-value store easily, so let's make a small table
        try:
            meta_table = self.db.create_table("meta", schema=pa.schema([
                pa.field("key", pa.string()),
                pa.field("value", pa.int64())
            ]), exist_ok=True)
            
            res = meta_table.search().where(f"key = '{key}'").limit(1).to_arrow()
            if len(res) > 0:
                return res["value"][0].as_py()
            return 0
        except Exception:
            return 0

    def _save_cursor(self, cursor: int):
        meta_table = self.db.open_table("meta")
        # Upsert logic? LanceDB doesn't support upsert by key easily yet in all versions.
        # We can delete and insert.
        meta_table.delete(f"key = 'source_cursor'")
        meta_table.add([{"key": "source_cursor", "value": cursor}])

    def _check_consistency(self):
        pass # LanceDB is atomic-ish

    def save_batch(self, records: List[Record], new_cursor: int):
        if not records: 
            self._save_cursor(new_cursor)
            return
            
        try:
            data = []
            for r in records:
                # 8-bit Quantization (Scaling)
                vec = r.vector
                if self.cfg.vector_dtype == "int8":
                    # Scale from [-1, 1] to [-127, 127]
                    vec = (vec * 127).clip(-128, 127).astype(np.int8)
                elif self.cfg.vector_dtype == "float16":
                    vec = vec.astype(np.float16)

                entry = {
                    "id": r.id,
                    "text": r.text,
                    "vector": vec,
                    # We store metadata as JSON string to handle arbitrary dicts
                    "metadata": json.dumps(r.metadata) if r.metadata else "{}",
                    "cursor": new_cursor # Associate batch with cursor
                }
                data.append(entry)
            
            self.table.add(data)
            self._save_cursor(new_cursor)
            self.current_id += len(records)
            
        except Exception as e:
            logger.error(f"Failed to save batch: {e}", exc_info=True)
            raise e

    def _save_cursor_only(self, new_cursor: int):
        self._save_cursor(new_cursor)

    def build_fts_index(self):
        if self.cfg.fts_mode == "none":
            return
            
        logger.info("Building vector index...")
        try:
            # We use L2 for int8/quantized search usually, or cosine if normalized
            # For int8, IVF_PQ or simple scalar index?
            # LanceDB will choose best based on data
            self.table.create_index(metric="cosine", vector_column_name="vector", replace=True)
        except Exception as e:
            logger.warning(f"Vector Index Build Error: {e} (might not be supported for {self.cfg.vector_dtype})")

        logger.info("Building FTS index (Tantivy)...")
        start_time = time.time()
        try:
            self.table.create_fts_index("text", replace=True)
            logger.info(f"FTS index ready in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"FTS Build Error: {e}")

    def load_hashes(self, processor: Optional[TextProcessor] = None) -> set[str]:
        # Warning: Loading all texts might be heavy for huge datasets.
        # Poorman-rag logic loaded all to memory.
        hashes = set()
        # Using iterator to avoid OOM
        try:
            # We select only 'text' column
            # Note: to_pandas() might load all. 
            # Use to_batches() for streaming.
            batch_iter = self.table.search().select(["text"]).limit(None).to_batches()
            
            for batch in batch_iter:
                texts = batch["text"]
                for raw_text in texts:
                    val = raw_text.as_py()
                    text = processor.clean(val) if processor else val
                    if text:
                        hashes.add(hashlib.md5(text.encode("utf-8")).hexdigest())
        except Exception as e:
            logger.warning(f"Could not load existing hashes (maybe empty table?): {e}")

        return hashes

    def optimize(self, older_than: timedelta = timedelta(0)):
        logger.info(f"Optimizing all LanceDB tables (cleaning up versions older than {older_than})...")
        start = time.time()
        
        # Get all table names
        table_names = self.db.table_names()
        for name in table_names:
            logger.info(f"  🧹 Optimizing table: {name}")
            table = self.db.open_table(name)
            table.compact_files()
            table.cleanup_old_versions(older_than=older_than, delete_unverified=True)
            
        logger.info(f"Optimization complete in {time.time() - start:.2f}s")

    def close(self):
        # We optimize with 0 retention for minimal file count before potential S3 sync
        self.optimize(older_than=timedelta(0))
        self.db.close()