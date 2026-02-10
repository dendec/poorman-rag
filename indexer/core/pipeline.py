import sys
import signal
import logging
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
        self.storage = Storage(config)
        self.processor = TextProcessor(
            chars_to_remove=config.chars_to_remove,
            normalization_mapping=config.normalization_mapping
        )
        pre_hashes = self.storage.load_hashes(self.processor)
        self.deduplicator = Deduplicator(pre_hashes)
        self.embedder = Embedder(config.model_name, compile_model=config.compile_model, max_length=config.max_tokens)
        
        # Load cursor (how many raw records we've already processed)
        self.processed_offset = self.storage.get_meta("source_cursor")
        
        self.batch = []          # Buffer for writing to DB (SQLite/USEARCH)
        self.gpu_buffer = []     # Buffer for sending to GPU (list of pairs: (text_for_emb, store_text, metadata))
        self.shutdown_requested = False
        
        # Signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if not self.shutdown_requested:
            logger.warning(f"Signal {signum} received. Graceful shutdown initiated...")
            self.shutdown_requested = True
        else:
            logger.critical("Forced exit requested.")
            sys.exit(1)

    def run(self):
        logger.info(f"Starting pipeline with batch_size={self.cfg.batch_size}")
        logger.info(f"Resuming from source offset: {self.processed_offset}")
        
        current_counter = 0 # Counter for "raw" records from the source
        
        try:
            with self.source as stream:
                total_items = len(self.source) 
                
                iterator = tqdm(
                    stream, 
                    desc="Indexing", 
                    total=total_items, 
                    initial=self.processed_offset
                )
                
                for item in iterator:
                    if self.shutdown_requested: break

                    # --- Data extraction ---
                    metadata = {}
                    if hasattr(item, "embedding_text"): # DataEntry object
                        text_for_embedding = item.embedding_text
                        text_for_storage   = item.storage_text
                        metadata           = getattr(item, "metadata", {})
                    elif isinstance(item, dict):
                        text_for_embedding = item.get("embedding_text", "")
                        text_for_storage   = item.get("storage_text", text_for_embedding)
                        metadata           = item.get("metadata", {})
                    else:
                        text_for_embedding = item
                        text_for_storage   = item

                    # --- Fast-Forward Logic ---
                    if current_counter < self.processed_offset:
                        current_counter += 1
                        continue
                    
                    current_counter += 1
                    
                    # --- Step 1: Cleaning ---
                    cleaned = self.processor.clean(text_for_embedding)
                    if not cleaned: continue
                    
                    # --- Step 2: Deduplication ---
                    if self.deduplicator.is_duplicate(cleaned): continue
                    
                    # --- Step 3: Token validation ---
                    n_tokens = self.embedder.token_count(text_for_embedding)
                    if not (self.cfg.min_tokens <= n_tokens <= self.cfg.max_tokens):
                        continue

                    # --- Step 4: Accumulation ---
                    self.gpu_buffer.append((text_for_embedding, text_for_storage, metadata))
                    
                    if len(self.gpu_buffer) >= self.cfg.batch_size:
                        self._process_gpu_batch(current_counter)

                # --- Finalization ---
                if self.gpu_buffer and not self.shutdown_requested:
                    self._process_gpu_batch(current_counter)
                
                if not self.shutdown_requested:
                    self.flush(new_cursor=current_counter)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)
        finally:
            self.close()

    def _process_gpu_batch(self, current_source_cursor: int):
        if not self.gpu_buffer:
            return

        logger.debug(f"Processing batch of {len(self.gpu_buffer)} items")
        embedding_texts = [self.cfg.prefix + item[0] for item in self.gpu_buffer]
        embeddings = self.embedder.embed(embedding_texts)
        
        for i, (emb_text, store_text, metadata) in enumerate(self.gpu_buffer):
            emb = embeddings[i]
            record_id = self.storage.current_id + len(self.batch)
            self.batch.append(Record(record_id, store_text, emb, metadata))

        self.gpu_buffer = []

        if len(self.batch) >= self.cfg.checkpoint_period:
            self.flush(new_cursor=current_source_cursor)

    def flush(self, new_cursor: int):
        """Saves batch and updates source position cursor"""
        if self.batch:
            logger.debug(f"Flushing {len(self.batch)} records to storage (cursor={new_cursor})")
            self.storage.save_batch(self.batch, new_cursor)
            self.batch = []
        else:
            self.storage._save_cursor_only(new_cursor)

    def close(self):
        if not self.shutdown_requested:
            if self.gpu_buffer:
                logger.warning("Emergency closing with unprocessed data in buffer.")
            
            logger.info("Building FTS5 search index...")
            self.storage.build_fts_index()
        
        self.storage.close()
        logger.info("Pipeline closed.")