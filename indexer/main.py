import argparse
import os
import sys
import logging
import importlib
from transformers import AutoTokenizer
from core.cfg import IndexingConfig
from core.datasource import FolderDataSource, DataSource
from core.pipeline import IndexingPipeline
from core.storage import Storage, Record
from datetime import timedelta
import boto3
from botocore.config import Config as BotoConfig


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
    if config.datasource:
        logger.info(f"🔌 Using custom datasource: {config.datasource}")
        try:
            full_path = config.datasource
            if ":" in full_path:
                module_name, class_name = full_path.split(":")
            else:
                module_name, class_name = full_path.rsplit(".", 1)
            
            module = importlib.import_module(module_name)
            CustomDataSource = getattr(module, class_name)
            
            # Instantiate custom source. We pass config and tokenizer often needed.
            # Assuming custom source constructor accepts (config=config, tokenizer=tokenizer) or (config, tokenizer)
            try:
                source = CustomDataSource(config=config, tokenizer=tokenizer)
            except TypeError:
                # Fallback to simple init if it doesn't accept args
                source = CustomDataSource()
                
        except Exception as e:
            logger.error(f"Failed to load datasource {config.datasource}: {e}")
            sys.exit(1)
    else:
        target_dir = args.dir or getattr(config, 'target_dir', None)
        if not target_dir:
            logger.error("Indexing directory not specified. Use --dir or set 'target_dir' in config.")
            sys.exit(1)
            
        logger.info(f"📁 Indexing directory: {target_dir}")
        source = FolderDataSource(target_dir, config, tokenizer)



    # Run Pipeline
    pipeline = IndexingPipeline(config, source)
    pipeline.run()
    logger.info("🎉 Indexing completed successfully!")

def handle_search(args, config, logger):
    """Starts interactive search loop."""
    logger.info("🔍 Initializing search engine...")
    search_engine = Search(config)
    search_engine.loop()

from core.s3 import get_uploader_from_config

def handle_sync(config, logger):
    """Syncs local index to S3."""
    uploader = get_uploader_from_config(config)
    if not uploader or not config.kb_alias:
        logger.error("❌ s3_bucket and kb_alias must be defined in config for sync.")
        sys.exit(1)

    source_dir = config.lancedb_uri
    if not os.path.exists(source_dir):
        logger.error(f"❌ Source directory {source_dir} does not exist.")
        sys.exit(1)

    prefix = f"rag/index/{config.kb_alias}"
    logger.info(f"📤 Syncing {source_dir} to s3://{config.s3_bucket}/{prefix}...")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_dir)
            s3_path = f"{prefix}/{relative_path}"
            
            uploader.upload_file(local_path, s3_path)

    logger.info("🎉 Sync completed successfully!")

def handle_optimize(config, logger):
    """Optimizes the LanceDB index."""
    logger.info(f"🧹 Optimizing index: {config.lancedb_uri}...")
    storage = Storage(config)
    # Perform aggressive optimization with 0 retention
    storage.optimize(older_than=timedelta(0))
    storage.close()
    logger.info("🎉 Optimization completed successfully!")

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

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync local index to S3")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize/Compact local index")

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
        elif args.action == "sync":
            handle_sync(config, logger)
        elif args.action == "optimize":
            handle_optimize(config, logger)

    except Exception as e:
        logger.critical(f"💥 Critical Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()