import argparse
import os
import sys
import logging
from typing import Any
import torch
from transformers import AutoTokenizer

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from core.cfg import IndexingConfig
from core.datasource import FolderDataSource
from core.pipeline import IndexingPipeline
from core.search import Search

def setup_logging(level_name: str):
    """Configures the logging system."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

import importlib

def get_datasource(config: IndexingConfig, tokenizer: Any) -> FolderDataSource:
    """Factory to get the appropriate datasource using reflection."""
    try:
        # Support both "module.submodule.Class" and "module.submodule:Class"
        path = config.datasource
        if ":" in path:
            module_path, class_name = path.split(":", 1)
        else:
            module_path, class_name = path.rsplit(".", 1)
            
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config, tokenizer)
    except Exception as e:
        raise RuntimeError(f"Failed to load datasource class '{config.datasource}': {e}") from e

def handle_index(args, config, logger):
    """Executes the indexing pipeline."""
    # Resolve model path: prefer local directory over HuggingFace hub
    slug = config.model_name.split("/")[-1]
    local_candidates = [
        config.model_name,
        f"models/{slug}",
        f"indexer/models/{slug}",
    ]
    resolved = config.model_name
    for candidate in local_candidates:
        if os.path.isdir(candidate):
            resolved = candidate
            logger.info(f"Using local model for tokenizer: {candidate}")
            break

    # Load tokenizer
    logger.info(f"🔌 Loading tokenizer for model: {resolved}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True, trust_remote_code=True)
    except Exception:
        logger.info(f"Local tokenizer not found, downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Setup DataSource
    if args.dir:
        config.target = args.dir
    
    try:
        source = get_datasource(config, tokenizer)
    except Exception as e:
        logger.error(f"❌ Failed to initialize datasource: {e}", exc_info=True)
        sys.exit(1)

    # Run Pipeline
    pipeline = IndexingPipeline(config, source)
    
    # Apply limit if specified
    if args.limit:
        logger.info(f"🛑 Limit applied: indexing only first {args.limit} items.")
        source_iter = iter(source)
        limited_source = (next(source_iter) for _ in range(args.limit))
        # We need a way to pass this to pipeline.run or modify run
        # For now, let's just modify the loop in pipeline.run if we can, 
        # but easier to just wrap the source in a limited iterator.
        
    pipeline.run(limit=args.limit)
    logger.info("🎉 Indexing completed successfully!")

def handle_search(args, config, logger):
    """Starts interactive search loop."""
    logger.info("🔍 Initializing search engine...")
    search_engine = Search(config)
    search_engine.loop()

def handle_rebuild_fts(args, config, logger):
    """Rebuilds the FTS5 index."""
    from core.storage import Storage
    logger.info(f"💾 Opening storage: {config.db_file}")
    storage = Storage(config)
    
    logger.info(f"🛠  Rebuilding FTS5 index in '{config.fts_mode}' mode...")
    storage.build_fts_index()
    
    logger.info("✨ Optimization (VACUUM)...")
    storage.optimize()
    
    db_size = os.path.getsize(config.db_file) / (1024 * 1024 * 1024)
    logger.info(f"✅ FTS5 rebuild completed! Final database size: {db_size:.2f} GB")

def handle_recall(args, config, logger):
    import numpy as np
    from tqdm import tqdm
    from core.search import Search
    
    logger.info(f"🧪 Starting Recall@{args.k} evaluation on {args.count} samples...")
    search_engine = Search(config)
    
    # Get random records
    cursor = search_engine.conn.execute(
        "SELECT id, text FROM dataset ORDER BY RANDOM() LIMIT ?", 
        (args.count,)
    )
    test_data = cursor.fetchall()
    
    if not test_data:
        logger.error("❌ Database is empty. Cannot run recall test.")
        return

    recalls = []
    logger.info(f"🔍 Processing samples...")
    
    for row in tqdm(test_data):
        doc_id, text = row['id'], row['text']
        if not text: continue
        
        # Use a random chunk of text as a query to make it a real semantic test
        text_len = len(text)
        if text_len <= 200:
            query = text
        else:
            import random
            start_idx = random.randint(0, text_len - 200)
            query = text[start_idx : start_idx + 200]
        
        # Vector search only
        try:
            prefixed_query = config.query_prefix + query
            vec = search_engine.embed.embed([prefixed_query])[0].flatten()
            matches = search_engine.index.search(vec, args.k)
            vector_ids = matches.keys.tolist()
            
            recalls.append(1.0 if doc_id in vector_ids else 0.0)
        except Exception as e:
            logger.error(f"Error during search for ID {doc_id}: {e}")

    if recalls:
        avg_recall = np.mean(recalls)
        logger.info(f"📈 Result: Average Recall@{args.k} = {avg_recall:.2%}")
    else:
        logger.error("❌ No successful tests performed.")

    search_engine.close()

def main():
    parser = argparse.ArgumentParser(description="poorman-rag Indexer CLI")
    
    # Global options
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )

    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents from a directory")
    index_parser.add_argument(
        "--dir", 
        type=str, 
        help="Path to the directory to index (overrides config)"
    )
    index_parser.add_argument(
        "--limit", 
        type=int, 
        help="Maximum number of items to index (for testing)"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Interactive search in the index")

    # Rebuild FTS command
    rebuild_parser = subparsers.add_parser("rebuild-fts", help="Rebuild FTS5 index from existing data")

    recall_parser = subparsers.add_parser("recall", help="Calculate Recall@K using random samples from DB")
    recall_parser.add_argument("--count", type=int, default=100, help="Number of random samples to test")
    recall_parser.add_argument("--k", type=int, default=50, help="Recall@K threshold")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("indexer")

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        # Load configuration
        logger.info(f"📂 Loading config from {args.config}...")
        config = IndexingConfig.from_yaml(args.config)
        
        if args.action == "index":
            handle_index(args, config, logger)
        elif args.action == "search":
            handle_search(args, config, logger)
        elif args.action == "rebuild-fts":
            handle_rebuild_fts(args, config, logger)
        elif args.action == "recall":
            handle_recall(args, config, logger)

    except Exception as e:
        logger.critical(f"💥 Critical Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()