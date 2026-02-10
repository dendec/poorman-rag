# poorman-rag Makefile

DIST_DIR=dist
BINARY_LAMBDA=$(DIST_DIR)/bootstrap
BINARY_MCP=$(DIST_DIR)/mcp
BINARY_EMBEDDING=$(DIST_DIR)/embedding
MCP_DEPLOY_ZIP=$(DIST_DIR)/mcp.zip
EMBEDDING_DEPLOY_ZIP=$(DIST_DIR)/embedding.zip
MAIN_MCP=cmd/mcp/main.go
MAIN_EMBEDDING=cmd/embedding/main.go
LIB_PATH=lib/linux_amd64/libonnxruntime.so

.PHONY: all build build-mcp build-embedding clean prepare-libs dist-dir test

all: build

dist-dir:
	@mkdir -p $(DIST_DIR)

prepare-libs:
	@if [ ! -f lib/linux_amd64/liblancedb_go.so ]; then \
		echo "📦 Decompressing LanceDB library..."; \
		gunzip -c lib/linux_amd64/liblancedb_go.so.gz > lib/linux_amd64/liblancedb_go.so; \
	fi

build: dist-dir prepare-libs build-mcp build-embedding

build-mcp: dist-dir prepare-libs
	@echo "🚀 Building Go binary for MCP server..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -ltokenizers -llancedb_go -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_MCP) $(MAIN_MCP)
	@echo "📦 Packaging into $(MCP_DEPLOY_ZIP)..."
	cp $(BINARY_MCP) $(BINARY_LAMBDA)
	zip -j $(MCP_DEPLOY_ZIP) $(BINARY_LAMBDA) lib/linux_amd64/liblancedb_go.so
	rm $(BINARY_LAMBDA)
	@echo "✅ MCP Lambda build complete: $(MCP_DEPLOY_ZIP)"

build-embedding: dist-dir prepare-libs
	@echo "🌐 Building embedding service binary..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -lonnxruntime -ltokenizers -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_EMBEDDING) $(MAIN_EMBEDDING)
	@echo "📦 Packaging into $(EMBEDDING_DEPLOY_ZIP)..."
	cp $(BINARY_EMBEDDING) $(BINARY_LAMBDA)
	zip -j $(EMBEDDING_DEPLOY_ZIP) $(BINARY_LAMBDA) lib/linux_amd64/libonnxruntime.so
	rm $(BINARY_LAMBDA)
	@echo "✅ Embedding Lambda build complete: $(EMBEDDING_DEPLOY_ZIP)"

test:
	@echo "🧪 Running unit tests (excluding CGO-dependent packages)..."
	go test ./internal/domain/... ./internal/services/... ./internal/utils/... ./internal/config/... -v
	@echo "✅ Unit tests completed"

test-integration: build-mcp
	@echo "🔍 Starting MCP Inspector for integration tests..."
	MODE=stdio npx @modelcontextprotocol/inspector -- ./$(BINARY_STDIO) --config indexer/config.yaml

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -f $(BINARY_LAMBDA) $(BINARY_STDIO) $(BINARY_EMBEDDING) $(MCP_DEPLOY_ZIP) $(EMBEDDING_DEPLOY_ZIP)
	@echo "✅ Cleanup complete"

clear:
	@echo "🗑️ Removing dist directory..."
	rm -rf $(DIST_DIR)
	@echo "✅ Dist directory cleared"
