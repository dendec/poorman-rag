# poorman-rag Makefile

BINARY_LAMBDA=bootstrap
BINARY_STDIO=mcp-stdio
DEPLOY_ZIP=deployment.zip
MAIN_LAMBDA=cmd/mcp/main.go
MAIN_STDIO=cmd/stdio/main.go
LIB_PATH=lib/libonnxruntime.so

.PHONY: all build build-lambda build-stdio clean

all: build

build: build-lambda build-stdio

build-lambda:
	@echo "🚀 Building Go binary for AWS Lambda (Linux AMD64)..."
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
	CGO_CFLAGS="-I$(PWD)/lib/include" \
	CGO_LDFLAGS="-L$(PWD)/lib" \
	go build -ldflags="-r \$$ORIGIN/lib" -o $(BINARY_LAMBDA) $(MAIN_LAMBDA)
	@echo "📦 Packaging into $(DEPLOY_ZIP)..."
	zip -j $(DEPLOY_ZIP) $(BINARY_LAMBDA) $(LIB_PATH)
	@echo "✅ Lambda build complete: $(DEPLOY_ZIP)"

build-stdio:
	@echo "💻 Building local Stdio binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(PWD)/lib/include" \
	CGO_LDFLAGS="-L$(PWD)/lib" \
	go build -ldflags="-r \$$ORIGIN/lib" -o $(BINARY_STDIO) $(MAIN_STDIO)
	@echo "✅ Stdio build complete: $(BINARY_STDIO)"

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -f $(BINARY_LAMBDA) $(BINARY_STDIO) $(DEPLOY_ZIP)
	@echo "✅ Cleanup complete"
