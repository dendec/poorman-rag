package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-lambda-go/lambda"
	"github.com/dendec/poorman-rag/internal/config"
	"github.com/dendec/poorman-rag/internal/infrastructure/embedding/loader"
	embedding_api "github.com/dendec/poorman-rag/internal/api/embedding"
)

var embeddingAPIAdapter *embedding_api.APIAdapter

func init() {
	InitSlog()
}

func initializeService() error {
	configPath := os.Getenv("CONFIG_PATH")
	cfg, err := config.Load(configPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Load the embedding service using the loader
	service, err := loader.LoadEmbeddingService(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize embedding service: %w", err)
	}

	// Create the API adapter
	embeddingAPIAdapter = embedding_api.NewAPIAdapter(service)

	return nil
}

// Lambda handler for AWS Lambda
func handleRequest(ctx context.Context, request events.APIGatewayV2HTTPRequest) (events.APIGatewayV2HTTPResponse, error) {
	if request.RequestContext.HTTP.Method != "POST" {
		return events.APIGatewayV2HTTPResponse{StatusCode: 405, Body: "Method Not Allowed"}, nil
	}

	var embeddingReq embedding_api.EmbeddingRequest
	if err := json.Unmarshal([]byte(request.Body), &embeddingReq); err != nil {
		slog.Error("invalid request body", "error", err)
		return events.APIGatewayV2HTTPResponse{StatusCode: 400, Body: "Invalid JSON"}, nil
	}

	response, err := embeddingAPIAdapter.HandleOpenAIRequest(ctx, embeddingReq)
	if err != nil {
		slog.Error("embedding request processing error", "error", err)
		return events.APIGatewayV2HTTPResponse{StatusCode: 500, Body: "Internal Error"}, nil
	}

	bodyBytes, _ := json.Marshal(response)

	return events.APIGatewayV2HTTPResponse{
		StatusCode: 200,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		Body: string(bodyBytes),
	}, nil
}

// HTTP handler for local server
func httpHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		slog.Error("failed to read request body", "error", err)
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var embeddingReq embedding_api.EmbeddingRequest
	if err := json.Unmarshal(body, &embeddingReq); err != nil {
		slog.Error("invalid request body", "error", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	response, err := embeddingAPIAdapter.HandleOpenAIRequest(ctx, embeddingReq)
	if err != nil {
		slog.Error("embedding request processing error", "error", err)
		http.Error(w, "Internal Error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CLI handler for command-line usage
func cliHandler(text string) error {
	ctx := context.Background()

	embedding, err := embeddingAPIAdapter.ComputeEmbedding(ctx, text)
	if err != nil {
		return fmt.Errorf("failed to compute embedding: %w", err)
	}

	// Output as JSON
	result := embedding_api.EmbeddingObj{
		Object:    "embedding",
		Index:     0,
		Embedding: embedding,
	}

	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	fmt.Println(string(output))
	return nil
}

func InitSlog() {
	var handler slog.Handler
	options := &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}

	if strings.ToLower(os.Getenv("LOG_LEVEL")) == "debug" {
		options.Level = slog.LevelDebug
	}

	if strings.ToLower(os.Getenv("LOG_FORMAT")) == "json" {
		handler = slog.NewJSONHandler(os.Stderr, options)
	} else {
		handler = slog.NewTextHandler(os.Stderr, options)
	}

	slog.SetDefault(slog.New(handler))
}

func main() {
	// Determine mode: Lambda, HTTP server, or CLI
	mode := os.Getenv("MODE")

	if mode == "lambda" {
		// AWS Lambda mode
		lambda.Start(handleRequest)
		return
	}

	if mode == "http" {
		// Initialize service for HTTP mode
		if err := initializeService(); err != nil {
			slog.Error("failed to initialize embedding service", "error", err)
			os.Exit(1)
		}

		// HTTP server mode
		port := os.Getenv("PORT")
		if port == "" {
			port = "8080"
		}

		slog.Info("starting HTTP server", "port", port)
		http.HandleFunc("/embeddings", httpHandler)
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/" && r.Method == "GET" {
				fmt.Fprintf(w, "Embedding Service\n")
				return
			}
			http.NotFound(w, r)
		})

		if err := http.ListenAndServe(":"+port, nil); err != nil {
			slog.Error("HTTP server error", "error", err)
			os.Exit(1)
		}
		return
	}

	// CLI mode - parse flags first to check for help
	textPtr := flag.String("text", "", "Text to embed")
	filePtr := flag.String("file", "", "File containing text to embed")
	configPtr := flag.String("config", "", "Path to config.yaml file")
	helpPtr := flag.Bool("help", false, "Show help message")
	hPtr := flag.Bool("h", false, "Show help message")

	flag.Parse()

	// Show help if requested
	if *helpPtr || *hPtr {
		showHelp()
		os.Exit(0)
	}

	// Set config path from flag if provided
	if *configPtr != "" {
		os.Setenv("CONFIG_PATH", *configPtr)
	}

	// For CLI mode, set log level to error only to avoid interfering with JSON output
	setCLILogLevel()

	// Initialize service for CLI mode
	if err := initializeService(); err != nil {
		slog.Error("failed to initialize embedding service", "error", err)
		os.Exit(1)
	}

	var text string
	if *textPtr != "" {
		text = *textPtr
	} else if *filePtr != "" {
		content, err := os.ReadFile(*filePtr)
		if err != nil {
			slog.Error("failed to read file", "error", err)
			os.Exit(1)
		}
		text = string(content)
	} else {
		// Read from stdin
		stdinData, err := io.ReadAll(bufio.NewReader(os.Stdin))
		if err != nil {
			slog.Error("failed to read from stdin", "error", err)
			os.Exit(1)
		}
		text = string(bytes.TrimSpace(stdinData))
	}

	if text == "" {
		showHelp()
		os.Exit(1)
	}

	if err := cliHandler(text); err != nil {
		slog.Error("CLI error", "error", err)
		os.Exit(1)
	}
}

// setCLILogLevel sets the log level to error only for CLI mode to avoid interfering with JSON output
func setCLILogLevel() {
	var handler slog.Handler
	options := &slog.HandlerOptions{
		Level: slog.LevelError, // Only show errors
	}

	if strings.ToLower(os.Getenv("LOG_FORMAT")) == "json" {
		handler = slog.NewJSONHandler(os.Stderr, options)
	} else {
		handler = slog.NewTextHandler(os.Stderr, options)
	}

	slog.SetDefault(slog.New(handler))
}

func showHelp() {
	fmt.Fprintln(os.Stderr, "Embedding CLI Tool")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  embedding-cli -text \"text to embed\"")
	fmt.Fprintln(os.Stderr, "  embedding-cli -file path/to/file.txt")
	fmt.Fprintln(os.Stderr, "  echo \"text to embed\" | embedding-cli")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Flags:")
	fmt.Fprintln(os.Stderr, "  -h, -help    Show this help message")
	fmt.Fprintln(os.Stderr, "  -text        Text to embed")
	fmt.Fprintln(os.Stderr, "  -file        File containing text to embed")
	fmt.Fprintln(os.Stderr, "  -config      Path to config.yaml file")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Configuration:")
	fmt.Fprintln(os.Stderr, "  Configuration is loaded from a YAML file specified with -config flag.")
	fmt.Fprintln(os.Stderr, "  See example config file for required fields.")
	fmt.Fprintln(os.Stderr, "")
}
