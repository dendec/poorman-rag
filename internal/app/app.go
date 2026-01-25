package app

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/dendec/poorman-rag/internal/config"
	"github.com/dendec/poorman-rag/internal/embedding"
	"github.com/dendec/poorman-rag/internal/embedding/onnx"
	"github.com/dendec/poorman-rag/internal/mcp"
	"github.com/dendec/poorman-rag/internal/search"
	"github.com/dendec/poorman-rag/internal/utils"
)

type App struct {
	Handler *mcp.Handler
	Manager *search.Manager
}

func NewApp(configPath string) (*App, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	if cfg.Bucket == "" || len(cfg.Aliases) == 0 {
		return nil, fmt.Errorf("missing RAG_BUCKET or RAG_KB_ALIASES in config/env")
	}

	isLocal := strings.ToLower(os.Getenv("REMOTE_EMBEDDING_ENABLED")) != "true"

	var (
		embedder embedding.Embedder
		dim      uint
	)

	if !isLocal {
		embedder, dim, err = initRemoteEmbedder()
	} else {
		embedder, dim, err = initLocalEmbedder(cfg)
	}

	if err != nil {
		return nil, fmt.Errorf("embedder init failed: %w", err)
	}

	slog.Info("initializing search manager", "aliases", cfg.Aliases)
	s3Opts := utils.S3Options{
		Bucket:    cfg.Bucket,
		Endpoint:  cfg.S3Endpoint,
		Region:    cfg.S3Region,
		AccessKey: cfg.S3AccessKey,
		SecretKey: cfg.S3SecretKey,
	}
	manager := search.NewManager(s3Opts, cfg.IndexDir, cfg.Aliases, dim, embedder)

	manager.SetRRFConfig(cfg.RRFK, cfg.LimitVector, cfg.LimitFTS, cfg.TopK)

	searchType := utils.StringFromEnvDefault("RAG_SEARCH_TYPE", mcp.SearchHybrid)
	handler := mcp.NewHandler(manager, searchType)

	return &App{
		Handler: handler,
		Manager: manager,
	}, nil
}

func initRemoteEmbedder() (embedding.Embedder, uint, error) {
	apiKey := utils.StringFromEnv("OPENAI_API_KEY")
	baseURL := utils.StringFromEnv("OPENAI_BASE_URL")
	apiModel := utils.StringFromEnv("MODEL")
	dim := uint(utils.IntFromEnv("DIMENSIONS"))

	slog.Info("initializing remote OpenAI-compatible embedder", "model", apiModel, "url", baseURL)
	return embedding.NewOpenAIEmbedder(apiKey, baseURL, apiModel), dim, nil
}

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

func initLocalEmbedder(cfg *config.Config) (embedding.Embedder, uint, error) {
	libPath := filepath.Join("lib", "libonnxruntime.so")

	// e.g. "multilingual-e5-small"
	slug := cfg.ModelName
	if parts := strings.Split(slug, "/"); len(parts) > 1 {
		slug = parts[len(parts)-1]
	}

	keys := getModelKeys(cfg.ModelName)
	finalPaths := make([]string, len(keys))

	toDownloadKeys := []string{}
	toDownloadIndices := []int{}

	// Check local "models/" directory first (where export_onnx.py saves by default)
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
			return nil, 0, err
		}
		for j, p := range downloadedPaths {
			finalPaths[toDownloadIndices[j]] = p
		}
	}

	// Paths order: 0:onnx, 1:tokenizer, 2:config
	onnxCfg, err := onnx.LoadConfig(finalPaths[2])
	if err != nil {
		return nil, 0, fmt.Errorf("model_config.json missing or invalid: %w", err)
	}
	onnxCfg.ModelPath = finalPaths[0]
	onnxCfg.TokenizerPath = finalPaths[1]

	slog.Info("initializing self-configuring local embedder",
		"model", cfg.ModelName,
		"dim", onnxCfg.Dimensions,
		"pooling", onnxCfg.Pooling,
	)

	emb, err := onnx.NewGenericEmbedder(libPath, onnxCfg)
	return emb, onnxCfg.Dimensions, err
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
