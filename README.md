# poorman-rag: Zero-Infrastructure Serverless MCP RAG

**poorman-rag** is a high-performance, cost-effective Retrieval-Augmented Generation (RAG) implementation designed to run entirely on **AWS Lambda** and **S3**. It natively supports the **Model Context Protocol (MCP)**, making it instantly compatible with AI agents like Claude Desktop and various IDE extensions.

## 🚀 Why poorman-rag?

*   **Zero Infrastructure**: No managed vector databases (Pinecone, Weaviate) required. Your data lives in S3 and compute happens in Lambda.
*   **Insanely Cost-Effective**: Pay only for what you use. Ideal for small to medium-sized knowledge bases where a $50/mo vector DB is overkill.
*   **Hybrid Search**: Combines **Vector Search** (semantic) with **SQLite FTS5** (keyword) using **Reciprocal Rank Fusion (RRF)** for superior result quality.
*   **Blazing Fast**: The search engine is written in **Go** using `usearch` and `onnxruntime` for local embedding inference and vector matching.
*   **Protocol Native**: Built from the ground up to support MCP, allowing any AI tool to query your knowledge base as a set of tools.

## 🏗️ Architecture

1.  **Indexer (Python)**: Processes your raw data, generates embeddings, and builds a compact SQLite + Usearch index.
2.  **S3 Storage**: Stores the compressed index files and model weights.
3.  **MCP Server (Go)**: A Lambda function that lazy-loads indexes from S3, performs hybrid search, and communicates via MCP.

## 🛠️ Quick Start

### 1. Prerequisites
*   AWS CLI configured
*   Serverless Framework installed (`npm install -g serverless`)
*   Go 1.21+ (for building the server)
*   Python 3.10+ (for indexing)

### 2. Index Your Data
Prepare your data using the provided Python scripts in the `indexer/` directory.
```bash
# Install dependencies
pip install -r indexer/requirements.txt

# Run the indexer (see docs/indexer.md for details)
python indexer/your_indexer.py index
```

### 3. Deploy to AWS
```bash
# Build the Go binary
GOOS=linux GOARCH=amd64 go build -o bootstrap cmd/mcp/main.go
zip deployment.zip bootstrap lib/libonnxruntime.so

# Deploy using Serverless
export RAG_BUCKET=your-bucket-name
export RAG_KB_ALIASES=docs
sls deploy
```

## 📜 Documentation

*   [Indexer Guide](docs/indexer.md) - How to prepare and upload your data.
*   [Deployment Guide](docs/deploy.md) - Detailed AWS setup and configuration.
*   [MCP Configuration](docs/mcp.md) - How to connect poorman-rag to your favorite AI agent.

## 🤝 Contributing

We welcome contributions! Please check our roadmap and open issues.

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details.
