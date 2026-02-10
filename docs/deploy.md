# Deployment Guide

poorman-rag is deployed as an AWS Lambda function with a Function URL.

## 1. Setup Environment Variables
Set these variables in your CI/CD or local shell before deploying:
- `RAG_BUCKET`: The S3 bucket where your indexes and models are stored.
- `RAG_KB_ALIASES`: Comma-separated list of knowledge base aliases (e.g., `docs,wiki`).
- `MODEL`: The embedding model name (default: `multilingual-e5-small`).
- `DIMENSIONS`: Embedding dimensions (default: `384`).

## 2. Build the Server
The simplest way to build the project is using the included `Makefile`, which handles all CGO flags and library linking correctly.

```bash
# Clean and build everything
make clean
make build
```

This creates a `bootstrap` binary and a `deployment.zip` containing the binary and necessary shared libraries (`libonnxruntime.so` and `liblancedb_go.so`) from `lib/linux_amd64/`.

## 3. Deploy
Ensure you have the Serverless Framework installed and configured with AWS credentials.

```bash
sls deploy --region us-east-1 --stage prod
```

## 4. Verify
After deployment, you will get a **Function URL**. You can test it with a POST request:
```bash
curl -X POST https://your-lambda-url/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```
