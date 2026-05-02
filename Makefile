# poorman-rag Makefile

BINARY_LAMBDA=bootstrap
BINARY_STDIO=mcp-stdio
DEPLOY_ZIP=deployment.zip
MAIN_LAMBDA=cmd/lambda/main.go
MAIN_STDIO=cmd/stdio/main.go
LIB_PATH=lib/libonnxruntime.so

.PHONY: all build build-lambda build-stdio clean

all: build

build: build-lambda build-stdio

build-lambda:
	@echo "🐳 Building/updating build image..."
	docker build -t poorman-rag-builder -f Dockerfile.build .
	@echo "🚀 Building Go binary for AWS Lambda..."
	docker run --rm \
		-v $(PWD):/app \
		-v $(shell go env GOMODCACHE):/go/pkg/mod \
		-w /app \
		poorman-rag-builder \
		"CGO_ENABLED=1 GOOS=linux GOARCH=amd64 GOMODCACHE=/go/pkg/mod \
		CGO_CFLAGS='-I/app/lib/include' \
		CGO_LDFLAGS='-L/app/lib' \
		go build -ldflags='-r \$$ORIGIN/lib' -o $(BINARY_LAMBDA) $(MAIN_LAMBDA)"
	@echo "📦 Packaging into $(DEPLOY_ZIP)..."
	rm -f $(DEPLOY_ZIP)
	zip $(DEPLOY_ZIP) $(BINARY_LAMBDA) config.yaml
	zip -r $(DEPLOY_ZIP) lib/*.so
	@echo "✅ Lambda build complete: $(DEPLOY_ZIP)"

build-stdio:
	@echo "💻 Building local Stdio binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I$(PWD)/lib/include" \
	CGO_LDFLAGS="-L$(PWD)/lib" \
	go build -ldflags="-r \$$ORIGIN/lib" -o $(BINARY_STDIO) $(MAIN_STDIO)
	@echo "✅ Stdio build complete: $(BINARY_STDIO)"


test: build-stdio
	@echo "🔍 Starting MCP Inspector..."
	npx @modelcontextprotocol/inspector -- ./$(BINARY_STDIO) --config indexer/config.yaml

deploy: build-lambda
	@echo "🚀 Deploying to AWS via Serverless (Stage: PROD)..."
	npx serverless deploy --stage prod

clean:
	@echo "🧹 Cleaning up build artifacts..."
	rm -f $(BINARY_LAMBDA) $(BINARY_STDIO) $(DEPLOY_ZIP)
	@echo "✅ Cleanup complete"
