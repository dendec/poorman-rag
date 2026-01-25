# Configuration Guide

`poorman-rag` uses a unified YAML configuration for indexing and automatically configures its Go server using metadata files.

## 1. Indexer Configuration (`config.yaml`)

### Storage Paths
- `db_file`: Path to the local SQLite database.
- `index_file`: Path to the local vector index file.
- `target_dir`: Directory containing documents to index.
- `extensions`: List of file extensions to include (e.g., `[".txt", ".md"]`).

### S3 / Cloud Storage (Multicloud)
Automate uploads after indexing by filling these fields:
- `s3_bucket`: Destination bucket name.
- `kb_alias`: Unique name for your Knowledge Base (used as a folder in S3).
- `s3_endpoint`: (Optional) Custom S3-compatible URL (e.g., `https://storage.googleapis.com` for GCP or R2 URL).
- `s3_region`: Cloud region (defaults to `us-east-1`).
- `s3_access_key` / `s3_secret_key`: (Optional) Credentials. If omitted, uses standard environment variables.

### Model Settings
- `model_name`: HuggingFace ID (e.g., `Qwen/Qwen3-Embedding-0.6B`).
- `max_tokens`: Context window size of the model.

### Chunks & Normalization
- `chunk_size`: Size of chunks in tokens.
- `chunk_overlap`: Overlap between chunks in tokens.
- `normalization_mapping`: Dict for text replacement during indexing.
- `chars_to_remove`: specific characters to strip from text.

---

## 2. Server Configuration (Environment Variables)

The Go server automatically downloads models and indexes. Configuration is minimal.

### Critical
- `RAG_BUCKET`: S3/Cloud bucket name.
- `RAG_KB_ALIASES`: Comma-separated list of active Knowledge Bases (e.g., `docs,support`).
- `MODEL`: Name of the model folder in S3 (e.g., `Qwen3-Embedding-0.6B`).

### Cloud Compatibility
- `RAG_S3_ENDPOINT`: Custom endpoint URL for non-AWS providers.
- `RAG_S3_REGION`: Region for S3 requests.

### Search Tuning (RRF)
- `RAG_RRF_K`: Smoothing constant for RRF (default: 60).
- `RAG_TOP_K`: Number of results to return (default: 5).
- `RAG_LIMIT_VECTOR`: Vector search results limit (default: 20).
- `RAG_LIMIT_FTS`: Full-text search results limit (default: 20).

---

## 3. Automation Scripts

### `export_onnx.py`
Exports and quantizes models to INT8. Inherits settings from `config.yaml`.
```bash
python scripts/export_onnx.py --config config.yaml --upload
```
- `--upload`: Automatically pushes exported files to your configured S3 bucket.
