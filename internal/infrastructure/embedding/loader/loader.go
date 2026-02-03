package loader

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/dendec/poorman-rag/internal/config"
	"github.com/dendec/poorman-rag/internal/embedding/onnx" // Use the original package
	embedding_service "github.com/dendec/poorman-rag/internal/services/embedding"
	"github.com/dendec/poorman-rag/internal/utils"
)

// LoadEmbeddingService creates an embedding service with the appropriate embedder
func LoadEmbeddingService(cfg *config.Config) (*embedding_service.Service, error) {
	libPath := findLibraryPath()
	if libPath == "" {
		return nil, fmt.Errorf("libonnxruntime.so not found in any of the expected locations")
	}

	// Get model files
	modelFiles, err := loadModelFiles(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to load model files: %w", err)
	}

	// Load ONNX config
	onnxCfg, err := onnx.LoadConfig(modelFiles[2]) // config is third file
	if err != nil {
		return nil, fmt.Errorf("model_config.json missing or invalid: %w", err)
	}
	onnxCfg.ModelPath = modelFiles[0]      // onnx model is first file
	onnxCfg.TokenizerPath = modelFiles[1]  // tokenizer is second file

	// Create ONNX embedder
	onnxEmbedder, err := onnx.NewGenericEmbedder(libPath, onnxCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX embedder: %w", err)
	}

	// Create and return the service
	service := embedding_service.NewService(onnxEmbedder, cfg.ModelName)
	return service, nil
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
		if err := utils.FileExists(path); err == nil {
			return path
		}
	}

	return ""
}

// loadModelFiles loads the required model files from local or S3
func loadModelFiles(cfg *config.Config) ([]string, error) {
	// Define the S3 keys for model files
	keys := getModelKeys(cfg.ModelName)
	if len(keys) == 0 {
		return nil, fmt.Errorf("model name is required to determine model files")
	}
	
	finalPaths := make([]string, len(keys))

	toDownloadKeys := []string{}
	toDownloadIndices := []int{}

	// Extract model slug from model name (e.g., "intfloat/multilingual-e5-small" -> "multilingual-e5-small")
	slug := cfg.ModelName
	if parts := splitModelName(cfg.ModelName); len(parts) > 1 {
		slug = parts[len(parts)-1]
	}

	// Check local "models/" directory first
	var localModelDir string

	// 1. Explicit config path (highest priority)
	if cfg.ModelsDir != "" {
		explicitPath := filepath.Join(cfg.ModelsDir, slug)
		if utils.DirExists(explicitPath) {
			localModelDir = explicitPath
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
			if utils.DirExists(dir) {
				localModelDir = dir
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
		if err := utils.FileExists(localPath); err == nil {
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

	return finalPaths, nil
}

// getModelKeys returns the S3 keys for model files
func getModelKeys(model string) []string {
	if model == "" {
		return nil
	}
	// If model is like "intfloat/multilingual-e5-small", we only want the slug "multilingual-e5-small"
	// for the S3 path, matching how export_onnx.py uploads it.
	parts := splitModelName(model)
	slug := parts[len(parts)-1]

	return []string{
		"rag/models/" + slug + "/model_quantized.onnx",
		"rag/models/" + slug + "/tokenizer.json",
		"rag/models/" + slug + "/model_config.json",
	}
}

// splitModelName splits a model name like "intfloat/multilingual-e5-small" into parts
func splitModelName(model string) []string {
	if model == "" {
		return nil
	}
	return strings.Split(model, "/")
}