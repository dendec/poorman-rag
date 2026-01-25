import argparse
import os
import sys
import logging
from transformers import AutoTokenizer
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

def handle_index(args, config, logger):
    """Executes the indexing pipeline."""
    # Load tokenizer
    logger.info(f"🔌 Loading tokenizer for model: {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Setup DataSource
    target_dir = args.dir or getattr(config, 'target_dir', None)
    if not target_dir:
        logger.error("Indexing directory not specified. Use --dir or set 'target_dir' in config.")
        sys.exit(1)
        
    logger.info(f"📁 Indexing directory: {target_dir}")
    source = FolderDataSource(target_dir, config, tokenizer)
    
    if len(source) == 0:
        logger.warning(f"No files found in {target_dir}")
        sys.exit(0)

    # Run Pipeline
    pipeline = IndexingPipeline(config, source)
    pipeline.run()
    logger.info("🎉 Indexing completed successfully!")

def handle_search(args, config, logger):
    """Starts interactive search loop."""
    logger.info("🔍 Initializing search engine...")
    search_engine = Search(config)
    search_engine.loop()

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

    # Search command
    search_parser = subparsers.add_parser("search", help="Interactive search in the index")

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

    except Exception as e:
        logger.critical(f"💥 Critical Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()