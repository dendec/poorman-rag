# Deployment Guide

poorman-rag is deployed as an AWS Lambda function with a Function URL.

## 1. Setup Environment Variables
Set these variables in your CI/CD or local shell before deploying:
- `RAG_BUCKET`: The S3 bucket where your indexes and models are stored.
- `RAG_KB_ALIASES`: Comma-separated list of knowledge base aliases (e.g., `docs,wiki`).
- `MODEL`: The embedding model name (default: `multilingual-e5-small`).
- `DIMENSIONS`: Embedding dimensions (default: `384`).

## 2. Build the Server
The server must be compiled for Linux AMD64 and bundled with the ONNX runtime library.

```bash
GOOS=linux GOARCH=amd64 go build -o bootstrap cmd/mcp/main.go
zip -j deployment.zip bootstrap lib/libonnxruntime.so
```

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
