package embedding_service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/dendec/poorman-rag/internal/config"
	"github.com/dendec/poorman-rag/internal/embedding"
	"github.com/dendec/poorman-rag/internal/embedding/onnx"
	"github.com/dendec/poorman-rag/internal/utils"
)

// EmbeddingRequest represents the OpenAI-compatible embedding request
type EmbeddingRequest struct {
	Input any    `json:"input"`           // Can be string, []string, or []int
	Model string `json:"model,omitempty"` // Model identifier
	User  string `json:"user,omitempty"`  // Optional user identifier
}

// EmbeddingResponse represents the OpenAI-compatible embedding response
type EmbeddingResponse struct {
	Object string         `json:"object"`
	Data   []EmbeddingObj `json:"data"`
	Model  string         `json:"model"`
	Usage  Usage          `json:"usage"`
}

// EmbeddingObj represents a single embedding result
type EmbeddingObj struct {
	Object    string      `json:"object"`
	Index     int         `json:"index"`
	Embedding []float32   `json:"embedding"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// Service handles embedding operations
type Service struct {
	embedder embedding.Embedder
	model    string
}

// NewService creates a new embedding service
func NewService(cfg *config.Config) (*Service, error) {
	// Use local ONNX model - this service is focused on local embedding only
	localEmbedder, err := initLocalEmbedder(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize local embedder: %w", err)
	}

	modelName := cfg.ModelName
	if modelName == "" {
		modelName = "unknown-model"
	}

	return &Service{
		embedder: localEmbedder,
		model:    modelName,
	}, nil
}

// initLocalEmbedder initializes a local ONNX embedder with S3 fallback
func initLocalEmbedder(cfg *config.Config) (embedding.Embedder, error) {
	// Look for the ONNX runtime library in multiple locations
	libPath := findLibraryPath()
	if libPath == "" {
		return nil, fmt.Errorf("libonnxruntime.so not found in any of the expected locations")
	}

	// Extract model slug from model name (e.g., "intfloat/multilingual-e5-small" -> "multilingual-e5-small")
	slug := cfg.ModelName
	if parts := strings.Split(slug, "/"); len(parts) > 1 {
		slug = parts[len(parts)-1]
	}

	// Define the S3 keys for model files
	keys := getModelKeys(cfg.ModelName)
	if len(keys) == 0 {
		return nil, fmt.Errorf("model name is required to determine model files")
	}
	finalPaths := make([]string, len(keys))

	toDownloadKeys := []string{}
	toDownloadIndices := []int{}

	// Check local "models/" directory first
	var localModelDir string

	// 1. Explicit config path (highest priority)
	if cfg.ModelsDir != "" {
		explicitPath := filepath.Join(cfg.ModelsDir, slug)
		if info, err := os.Stat(explicitPath); err == nil && info.IsDir() {
			localModelDir = explicitPath
			slog.Info("using configured local model directory", "path", localModelDir)
		} else {
			slog.Warn("configured models_dir not found or invalid", "path", explicitPath)
		}
	}

	// 2. Heuristic search (if explicit not matched)
	if localModelDir == "" {
		// We search in a few common locations relative to CWD
		candidates := []string{
			filepath.Join("models", slug),
			filepath.Join("indexer", "models", slug),
			filepath.Join("..", "models", slug),
		}

		for _, dir := range candidates {
			if info, err := os.Stat(dir); err == nil && info.IsDir() {
				localModelDir = dir
				slog.Info("found local model directory", "path", dir)
				break
			}
		}
	}

	// If not found, we will check "models/slug" but likely fail validation and trigger download to /tmp
	if localModelDir == "" {
		localModelDir = filepath.Join("models", slug)
	}

	// Check if local files exist, otherwise queue for download
	for i, key := range keys {
		fileName := filepath.Base(key)
		localPath := filepath.Join(localModelDir, fileName)

		// Simple existence check
		if info, err := os.Stat(localPath); err == nil && !info.IsDir() && info.Size() > 0 {
			slog.Info("using local model file", "path", localPath)
			finalPaths[i] = localPath
		} else {
			toDownloadKeys = append(toDownloadKeys, key)
			toDownloadIndices = append(toDownloadIndices, i)
		}
	}

	// Download missing files from S3
	if len(toDownloadKeys) > 0 {
		opts := utils.S3Options{
			Bucket:    cfg.Bucket,
			Endpoint:  cfg.S3Endpoint,
			Region:    cfg.S3Region,
			AccessKey: cfg.S3AccessKey,
			SecretKey: cfg.S3SecretKey,
		}
		downloadedPaths, err := utils.EnsureDownload(opts, toDownloadKeys...)
		if err != nil {
			return nil, fmt.Errorf("failed to download model files from S3: %w", err)
		}
		for j, p := range downloadedPaths {
			finalPaths[toDownloadIndices[j]] = p
		}
	}

	// Validate that all required files are present
	for i, path := range finalPaths {
		if path == "" {
			return nil, fmt.Errorf("missing required model file: %s", keys[i])
		}
	}

	// Paths order: 0:onnx, 1:tokenizer, 2:config
	onnxCfg, err := onnx.LoadConfig(finalPaths[2])
	if err != nil {
		return nil, fmt.Errorf("model_config.json missing or invalid: %w", err)
	}
	onnxCfg.ModelPath = finalPaths[0]
	onnxCfg.TokenizerPath = finalPaths[1]

	slog.Info("initializing self-configuring local embedder",
		"model", cfg.ModelName,
		"dim", onnxCfg.Dimensions,
		"pooling", onnxCfg.Pooling,
	)

	return onnx.NewGenericEmbedder(libPath, onnxCfg)
}

// getModelKeys returns the S3 keys for model files
func getModelKeys(model string) []string {
	if model == "" {
		return nil
	}
	// If model is like "intfloat/multilingual-e5-small", we only want the slug "multilingual-e5-small"
	// for the S3 path, matching how export_onnx.py uploads it.
	parts := strings.Split(model, "/")
	slug := parts[len(parts)-1]

	return []string{
		"rag/models/" + slug + "/model_quantized.onnx",
		"rag/models/" + slug + "/tokenizer.json",
		"rag/models/" + slug + "/model_config.json",
	}
}

// findLibraryPath looks for the ONNX runtime library in multiple locations
func findLibraryPath() string {
	// Possible locations to search for the library
	locations := []string{
		"libonnxruntime.so",                    // Current directory
		filepath.Join("lib", "libonnxruntime.so"), // lib/ subdirectory
		filepath.Join("..", "lib", "libonnxruntime.so"), // Parent's lib/ subdirectory
		filepath.Join("lib", "linux_amd64", "libonnxruntime.so"), // lib/linux_amd64/ subdirectory
		filepath.Join("..", "lib", "linux_amd64", "libonnxruntime.so"), // Parent's lib/linux_amd64/ subdirectory
	}

	for _, path := range locations {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}

// ComputeEmbedding computes embeddings for a single text
func (s *Service) ComputeEmbedding(ctx context.Context, text string) ([]float32, error) {
	return s.embedder.Embed(ctx, text)
}

// HandleOpenAIRequest handles OpenAI-compatible embedding requests
func (s *Service) HandleOpenAIRequest(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	var texts []string

	// Parse input - can be string, []string, or []int
	switch v := req.Input.(type) {
	case string:
		texts = []string{v}
	case []interface{}:
		for _, item := range v {
			switch val := item.(type) {
			case string:
				texts = append(texts, val)
			case float64: // JSON numbers are parsed as float64
				texts = append(texts, fmt.Sprintf("%.0f", val)) // Convert to string representation
			default:
				return nil, fmt.Errorf("unsupported input type in array: %T", val)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported input type: %T", v)
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("no input provided")
	}

	// Compute embeddings for all texts
	embeddings := make([]EmbeddingObj, len(texts))
	for i, text := range texts {
		embedding, err := s.ComputeEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to compute embedding for text %d: %w", i, err)
		}

		embeddings[i] = EmbeddingObj{
			Object:    "embedding",
			Index:     i,
			Embedding: embedding,
		}
	}

	// Calculate usage stats (simple approximation)
	totalTokens := 0
	for _, text := range texts {
		totalTokens += len(strings.Fields(text)) // Rough token count
	}

	response := &EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  s.model,
		Usage: Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	return response, nil
}

// HTTPHandler returns an HTTP handler for the embedding API
func (s *Service) HTTPHandler(w http.ResponseWriter, r *http.Request) {
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

	var embeddingReq EmbeddingRequest
	if err := json.Unmarshal(body, &embeddingReq); err != nil {
		slog.Error("invalid request body", "error", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	response, err := s.HandleOpenAIRequest(ctx, embeddingReq)
	if err != nil {
		slog.Error("embedding request processing error", "error", err)
		http.Error(w, "Internal Error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}