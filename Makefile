# poorman-rag Makefile

BINARY_LAMBDA=bootstrap
BINARY_STDIO=mcp-stdio
DEPLOY_ZIP=deployment.zip
MAIN_LAMBDA=cmd/mcp/main.go
MAIN_STDIO=cmd/stdio/main.go
LIB_PATH=lib/linux_amd64/libonnxruntime.so

.PHONY: all build build-lambda build-stdio clean prepare-libs

all: build

prepare-libs:
	@if [ ! -f lib/linux_amd64/liblancedb_go.so ]; then \
		echo "📦 Decompressing LanceDB library..."; \
		gunzip -c lib/linux_amd64/liblancedb_go.so.gz > lib/linux_amd64/liblancedb_go.so; \
	fi

build: prepare-libs build-lambda build-stdio

build-lambda: prepare-libs
	@echo "🚀 Building Go binary for AWS Lambda (Linux AMD64)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -ltokenizers -llancedb_go -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_LAMBDA) $(MAIN_LAMBDA)
	@echo "📦 Packaging into $(DEPLOY_ZIP)..."
	# Include both ONNX and LanceDB shared libs if they exist
	zip -j $(DEPLOY_ZIP) $(BINARY_LAMBDA) lib/linux_amd64/libonnxruntime.so lib/linux_amd64/liblancedb_go.so
	@echo "✅ Lambda build complete: $(DEPLOY_ZIP)"

build-stdio: prepare-libs
	@echo "💻 Building local Stdio binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(PWD)/lib/include -I$(PWD)/include" \
	CGO_LDFLAGS="-Wl,--allow-multiple-definition -L$(PWD)/lib/linux_amd64 -ltokenizers -llancedb_go -lm -ldl -lpthread -lstdc++" \
	go build -ldflags="-r \$$ORIGIN/lib/linux_amd64" -o $(BINARY_STDIO) $(MAIN_STDIO)
	@echo "✅ Stdio build complete: $(BINARY_STDIO)"


test: build-stdio
	@echo "🔍 Starting MCP Inspector..."
	npx @modelcontextprotocol/inspector -- ./$(BINARY_STDIO) --config indexer/config.yaml

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -f $(BINARY_LAMBDA) $(BINARY_STDIO) $(DEPLOY_ZIP)
	@echo "✅ Cleanup complete"
