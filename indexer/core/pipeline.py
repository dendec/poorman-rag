import sys
import os
import json
import signal
import logging
from typing import Optional, Dict, List
from tqdm import tqdm
from core.cfg import IndexingConfig
from core.datasource import DataSource
from core.deduplicator import Deduplicator, TextProcessor
from core.embedder import Embedder
from core.storage import Storage, Record

logger = logging.getLogger("indexer.pipeline")

class IndexingPipeline:
    def __init__(self, config: IndexingConfig, source: DataSource):
        self.source = source
        self.cfg = config
        self.multi_index = bool(config.output_dir)

        self.processor = TextProcessor(
            chars_to_remove=config.chars_to_remove,
            normalization_mapping=config.normalization_mapping
        )

        if self.multi_index:
            os.makedirs(config.output_dir, exist_ok=True)
            self.storages: Dict[str, Storage] = {}
            self.forum_names: Dict[str, str] = {}
            self.batches: Dict[str, List[Record]] = {}
            self._cursor_file = os.path.join(config.output_dir, "_cursor.json")
            self.processed_offset = self._load_cursor()
            pre_hashes = self._load_all_hashes()
        else:
            self.storage = Storage(config)
            self.batch: List[Record] = []
            self.processed_offset = self.storage.get_meta("source_cursor")
            pre_hashes = self.storage.load_hashes(self.processor)

        self.deduplicator = Deduplicator(pre_hashes)
        self.embedder = Embedder.from_config(config)

        self.gpu_buffer = []  # (emb_text, store_text, fts_text, metadata, index_key)
        self.shutdown_requested = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Multi-index helpers ────────────────────────────────────────────────

    def _load_cursor(self) -> int:
        if os.path.exists(self._cursor_file):
            with open(self._cursor_file) as f:
                return json.load(f).get("source_cursor", 0)
        return 0

    def _save_cursor(self, cursor: int):
        with open(self._cursor_file, "w") as f:
            json.dump({"source_cursor": cursor}, f)

    def _load_all_hashes(self) -> set:
        hashes = set()
        if not os.path.exists(self.cfg.output_dir):
            return hashes
        for key in os.listdir(self.cfg.output_dir):
            db_path = os.path.join(self.cfg.output_dir, key, "dataset.sqlite")
            if os.path.exists(db_path):
                tmp = Storage(self.cfg, db_file=db_path, index_file=os.path.join(self.cfg.output_dir, key, "vectors.usearch"))
                hashes.update(tmp.load_hashes(self.processor))
                tmp.close()
        return hashes

    def _get_storage(self, key: str, forum_name: str) -> Storage:
        if key not in self.storages:
            # Safety cap: keep at most 2000 open storages (more than any real dataset's forum count).
            # This prevents infinite memory growth if the index_key domain is huge.
            if len(self.storages) >= 2000:
                # Close the first one (oldest)
                oldest_key = next(iter(self.storages))
                logger.debug(f"Evicting storage for forum {oldest_key} (LRU)")
                self.storages[oldest_key].flush()  # lightweight: no VACUUM/compression
                del self.storages[oldest_key]
                self.batches.pop(oldest_key, None)

            db_dir = os.path.join(self.cfg.output_dir, key)
            os.makedirs(db_dir, exist_ok=True)
            
            db_file = os.path.join(db_dir, "dataset.sqlite")
            index_file = os.path.join(db_dir, "vectors.usearch")
            
            storage = Storage(self.cfg, db_file=db_file, index_file=index_file)
            storage.save_meta_str("forum_name", forum_name)
            self.storages[key] = storage
            self.forum_names[key] = forum_name
            self.batches[key] = []
            
        return self.storages[key]

    # ── Signal handling ────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        if not self.shutdown_requested:
            logger.warning(f"Signal {signum} received. Graceful shutdown initiated...")
            self.shutdown_requested = True
        else:
            logger.critical("Forced exit requested.")
            sys.exit(1)

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self, limit: Optional[int] = None):
        import threading
        import queue as queue_module

        logger.info(f"Starting pipeline (multi_index={self.multi_index}, batch_size={self.cfg.batch_size})")
        logger.info(f"Resuming from source offset: {self.processed_offset}")
        
        # Queue holds ready-to-embed batches for the GPU thread.
        # Maxsize=2 means CPU can pre-prepare at most 1 extra batch ahead.
        batch_queue: queue_module.Queue = queue_module.Queue(maxsize=2)
        _SENTINEL = object()  # signals producer is done

        def producer():
            """CPU thread: parse, clean, deduplicate, accumulate, enqueue batches."""
            current_counter = self.processed_offset
            indexed_count = 0
            buffer = []

            try:
                with self.source as stream:
                    # Tell the source to skip already processed items if it can
                    self.source.skip(self.processed_offset)
                    
                    total_items = len(self.source)
                    with tqdm(desc="Indexing", total=total_items, initial=self.processed_offset) as pbar:
                        for item in stream:
                            if self.shutdown_requested:
                                break
                            if limit and indexed_count >= limit:
                                break

                            # --- Data extraction ---
                            if hasattr(item, "embedding_text"):
                                text_for_embedding = item.embedding_text
                                text_for_storage   = item.text
                                fts_text           = getattr(item, "fts_text", None)
                                metadata           = getattr(item, "metadata", {})
                                index_key          = getattr(item, "index_key", None)
                            elif isinstance(item, dict):
                                text_for_embedding = item.get("embedding_text", "")
                                text_for_storage   = item.get("storage_text", text_for_embedding)
                                fts_text           = item.get("fts_text", None)
                                metadata           = item.get("metadata", {})
                                index_key          = item.get("index_key", None)
                            else:
                                text_for_embedding = item
                                text_for_storage   = item
                                fts_text = None
                                metadata = {}
                                index_key = None

                            # --- Fast-Forward ---
                            if current_counter < self.processed_offset:
                                current_counter += 1
                                continue
                            current_counter += 1
                            pbar.update(1)

                            # --- Step 1: Cleaning ---
                            cleaned = self.processor.clean(text_for_embedding)
                            if not cleaned:
                                continue

                            # --- Step 2: Deduplication ---
                            if self.deduplicator.is_duplicate(cleaned):
                                continue

                            # --- Step 3: Token validation ---
                            n_tokens = self.embedder.token_count(text_for_embedding)
                            if n_tokens < self.cfg.min_tokens:
                                continue

                            # --- Step 4: Accumulate ---
                            buffer.append((text_for_embedding, text_for_storage, fts_text, metadata, index_key))
                            indexed_count += 1

                            if len(buffer) >= self.cfg.batch_size:
                                batch_queue.put((buffer, current_counter))
                                buffer = []

                    # Enqueue remaining items
                    if buffer:
                        batch_queue.put((buffer, current_counter))
            except Exception as e:
                logger.error(f"Producer thread failed: {e}", exc_info=True)
            finally:
                batch_queue.put(_SENTINEL)

        producer_thread = threading.Thread(target=producer, daemon=True, name="IndexProducer")
        producer_thread.start()

        last_cursor = 0
        try:
            while True:
                item = batch_queue.get()
                if item is _SENTINEL:
                    break
                batch, cursor = item
                last_cursor = cursor
                self.gpu_buffer = batch
                self._process_gpu_batch(cursor)

            if not self.shutdown_requested:
                self.flush(new_cursor=last_cursor)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.shutdown_requested = True
            sys.exit(1)
        finally:
            producer_thread.join(timeout=5)
            self.close()


    # ── Batch processing ───────────────────────────────────────────────────

    def _process_gpu_batch(self, current_source_cursor: int):
        if not self.gpu_buffer:
            return

        logger.debug(f"Processing batch of {len(self.gpu_buffer)} items")
        embedding_texts = [self.cfg.prefix + item[0] for item in self.gpu_buffer]
        embeddings = self.embedder.embed(embedding_texts)

        for i, (emb_text, store_text, fts_text, metadata, index_key) in enumerate(self.gpu_buffer):
            emb = embeddings[i]

            if self.multi_index:
                key = index_key or "_default"
                forum_name = metadata.get("forum", key) if metadata else key
                storage = self._get_storage(key, forum_name)
                record_id = storage.current_id + len(self.batches[key])
                self.batches[key].append(Record(record_id, emb, store_text, fts_text, metadata))
            else:
                record_id = self.storage.current_id + len(self.batch)
                self.batch.append(Record(record_id, emb, store_text, fts_text, metadata))

        self.gpu_buffer = []

        total_buffered = sum(len(b) for b in self.batches.values()) if self.multi_index else len(self.batch)
        if total_buffered >= self.cfg.checkpoint_period:
            self.flush(new_cursor=current_source_cursor)

    # ── Flush & Close ──────────────────────────────────────────────────────

    def flush(self, new_cursor: int):
        if self.multi_index:
            for key, batch in self.batches.items():
                if batch:
                    logger.info(f"Flushing {len(batch)} records to index '{key}' (cursor={new_cursor})")
                    self.storages[key].save_batch(batch, new_cursor)
                    self.batches[key] = []
            self._save_cursor(new_cursor)
        else:
            if self.batch:
                logger.info(f"Flushing {len(self.batch)} records to storage (cursor={new_cursor})")
                self.storage.save_batch(self.batch, new_cursor)
                self.batch = []
            else:
                self.storage._save_cursor_only(new_cursor)

    def close(self):
        if self.multi_index:
            for key, storage in list(self.storages.items()):
                count = storage.current_id
                if count < self.cfg.min_index_entries:
                    logger.warning(f"Dropping index '{key}' ({count} entries < min={self.cfg.min_index_entries})")
                    storage.conn.close()
                    # Remove the directory
                    import shutil
                    key_dir = os.path.join(self.cfg.output_dir, key)
                    shutil.rmtree(key_dir, ignore_errors=True)
                    del self.storages[key]
                    self.forum_names.pop(key, None)
                    continue

                if not self.shutdown_requested:
                    logger.info(f"Building FTS5 index for '{key}'...")
                    storage.build_fts_index()
                storage.close()

            # Write global forums_map.json
            map_path = os.path.join(self.cfg.output_dir, "forums_map.json")
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(self.forum_names, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 Forums map written to {map_path} ({len(self.forum_names)} forums)")
        else:
            if not self.shutdown_requested:
                if self.gpu_buffer:
                    logger.warning("Emergency closing with unprocessed data in buffer.")
                logger.info("Building FTS5 search index...")
                self.storage.build_fts_index()
            self.storage.close()

        logger.info("Pipeline closed.")