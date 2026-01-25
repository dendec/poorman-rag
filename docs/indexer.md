# Indexer Guide

This guide explains how to prepare your knowledge base for poorman-rag.

## 1. Configuration
The indexer uses a YAML configuration file. You can find a template in `indexer/config.yaml.example`.

### Key Parameters:
- `model_name`: HuggingFace model ID for generating embeddings (e.g., `intfloat/multilingual-e5-small`).
- `vector_dim`: Must match your model (384 for e5-small, 768 for Gemma).
- `max_tokens`: Maximum context size of the model. 
- `chunk_size`: Size of text chunks in units of **tokens**.
- `chunk_overlap`: Number of overlapping tokens between chunks.
- `fts_mode`: `size` (compact) or `speed` (fast lookup, larger index).
- `target_dir`: Path to the directory containing your documents.
- `extensions`: List of file extensions to index (e.g., `[".txt", ".md"]`).

## 2. Building the Index
Run the unified indexer CLI with the `index` action:
```bash
python indexer/main.py --config indexer/config.yaml index
```
*Note: You can override the directory from the config using `--dir`:*
```bash
python indexer/main.py --config indexer/config.yaml index --dir ./my_docs
```

## 3. Testing the Index
You can test your index locally using the `search` action. This starts an interactive CLI loop:
```bash
python indexer/main.py --config indexer/config.yaml search
```
▶ `query >` *your search term*

## 4. Exporting the Model to ONNX
poorman-rag uses ONNX for local inference in Lambda.
```bash
python indexer/scripts/export_onnx.py --model_id intfloat/multilingual-e5-small
```

## 5. Upload to S3
Upload the generated files to your S3 bucket:
```bash
aws s3 cp content.sqlite.zst s3://$RAG_BUCKET/rag/index/$ALIAS/
aws s3 cp vectors.usearch s3://$RAG_BUCKET/rag/index/$ALIAS/
aws s3 cp model_quantized.onnx s3://$RAG_BUCKET/rag/models/$MODEL/
aws s3 cp tokenizer.json s3://$RAG_BUCKET/rag/models/$MODEL/
```
