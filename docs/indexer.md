# Indexer Guide

This guide explains how to prepare your knowledge base for poorman-rag.

## 1. Configuration
The indexer uses a YAML configuration file. You can find a template in `indexer/config.yaml.example`.

### Key Parameters:
- `lancedb_uri`: Path to local LanceDB index (e.g., `index/lancedb`).
- `table_name`: Name of the table within LanceDB (default: `dataset`).
- `s3_bucket`: Destination bucket for search indices and models.
- `kb_alias`: Alias for this knowledge base (used in S3 paths).

## 2. Building the Index
Run the unified indexer CLI with the `index` action. This will automatically run `optimize` at the end:
```bash
python indexer/main.py --config indexer/config.yaml index
```

## 3. Optimizing the Index
You can manually trigger compaction to merge fragments and delete old versions:
```bash
python indexer/main.py --config indexer/config.yaml optimize
```

## 4. Testing the Index
You can test your index locally using the `search` action. This starts an interactive CLI loop:
```bash
python indexer/main.py --config indexer/config.yaml search
```

## 5. Exporting the Model to ONNX
poorman-rag uses ONNX for local inference in Lambda.
```bash
python indexer/main.py --config indexer/config.yaml export-onnx
```

## 6. Syncing to S3
The indexer now includes a built-in sync command that uses your configuration to upload the index to the correct S3 location:
```bash
python indexer/main.py --config indexer/config.yaml sync
```

This replaces manual `aws s3 cp` commands and ensures the directory structure matches Go runtime expectations.
