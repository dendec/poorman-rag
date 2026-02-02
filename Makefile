# poorman-rag Makefile

DIST_DIR=dist
BINARY_LAMBDA=$(DIST_DIR)/bootstrap
BINARY_STDIO=$(DIST_DIR)/mcp-stdio
BINARY_EMBEDDING=$(DIST_DIR)/embedding-bootstrap
MCP_DEPLOY_ZIP=$(DIST_DIR)/mcp.zip
EMBEDDING_DEPLOY_ZIP=$(DIST_DIR)/embedding.zip
MAIN_LAMBDA=cmd/mcp/main.go
MAIN_STDIO=cmd/stdio/main.go
MAIN_EMBEDDING=cmd/embedding/main.go
LIB_PATH=lib/linux_amd64/libonnxruntime.so

.PHONY: all build build-lambda build-stdio build-embedding clean prepare-libs dist-dir

all: build

dist-dir:
	@mkdir -p $(DIST_DIR)

prepare-libs:
	@if [ ! -f lib/linux_amd64/liblancedb_go.so ]; then \
		echo "📦 Decompressing LanceDB library..."; \
		gunzip -c lib/linux_amd64/liblancedb_go.so.gz > lib/linux_amd64/liblancedb_go.so; \
	fi

build: dist-dir prepare-libs build-lambda build-stdio build-embedding

build-lambda: dist-dir prepare-libs
	@echo "🚀 Building Go binary for AWS Lambda (Linux AMD64)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -ltokenizers -llancedb_go -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_LAMBDA) $(MAIN_LAMBDA)
	@echo "📦 Packaging into $(MCP_DEPLOY_ZIP)..."
	# Include both ONNX and LanceDB shared libs if they exist
	zip -j $(MCP_DEPLOY_ZIP) $(BINARY_LAMBDA) lib/linux_amd64/libonnxruntime.so lib/linux_amd64/liblancedb_go.so
	@echo "✅ Lambda build complete: $(MCP_DEPLOY_ZIP)"

build-stdio: dist-dir prepare-libs
	@echo "💻 Building local Stdio binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -ltokenizers -llancedb_go -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_STDIO) $(MAIN_STDIO)
	@echo "✅ Stdio build complete: $(BINARY_STDIO)"

build-embedding: dist-dir prepare-libs
	@echo "🌐 Building embedding service binary for AWS Lambda (Linux AMD64)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -lonnxruntime -ltokenizers -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_EMBEDDING) $(MAIN_EMBEDDING)
	@echo "📦 Packaging into $(EMBEDDING_DEPLOY_ZIP)..."
	# Lambda runtime expects the executable to be named 'bootstrap'
	cp $(BINARY_EMBEDDING) bootstrap
	zip -j $(EMBEDDING_DEPLOY_ZIP) bootstrap lib/linux_amd64/libonnxruntime.so
	rm bootstrap
	@echo "✅ Embedding Lambda build complete: $(EMBEDDING_DEPLOY_ZIP)"

test: build-stdio
	@echo "🔍 Starting MCP Inspector..."
	npx @modelcontextprotocol/inspector -- ./$(BINARY_STDIO) --config indexer/config.yaml

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -f $(BINARY_LAMBDA) $(BINARY_STDIO) $(BINARY_EMBEDDING) $(MCP_DEPLOY_ZIP) $(EMBEDDING_DEPLOY_ZIP)
	@echo "✅ Cleanup complete"
