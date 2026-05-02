package app

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/dendec/poorman-rag/internal/config"
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
		return nil, fmt.Errorf("missing s3_bucket or knowledge bases (kb) in config")
	}

	slog.Info("initializing search manager", "aliases", cfg.Aliases)
	s3Opts := utils.S3Options{
		Bucket:    cfg.Bucket,
		Endpoint:  cfg.S3Endpoint,
		Region:    cfg.S3Region,
		AccessKey: cfg.S3AccessKey,
		SecretKey: cfg.S3SecretKey,
	}
	manager := search.NewManager(s3Opts, cfg.IndexDir, cfg.ModelDir, cfg.KBs)

	searchType := utils.StringFromEnvDefault("RAG_SEARCH_TYPE", mcp.SearchHybrid)
	handler := mcp.NewHandler(manager, searchType)

	return &App{
		Handler: handler,
		Manager: manager,
	}, nil
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
