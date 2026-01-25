package config

import (
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	// S3 / Cloud Storage
	Bucket      string `yaml:"s3_bucket"`
	KBAlias     string `yaml:"kb_alias"` // Single alias in config.yaml
	S3Endpoint  string `yaml:"s3_endpoint"`
	S3Region    string `yaml:"s3_region"`
	S3AccessKey string `yaml:"s3_access_key"`
	S3SecretKey string `yaml:"s3_secret_key"`

	// Model
	ModelName string `yaml:"model_name"`
	ModelsDir string `yaml:"models_dir"` // Optional explicit path to models directory
	IndexDir  string `yaml:"index_dir"`  // Optional explicit path to index directory

	// Search Tuning
	RRFK        float64 `yaml:"rrf_k"`
	LimitVector int     `yaml:"limit_vector"`
	LimitFTS    int     `yaml:"limit_fts"`
	TopK        int     `yaml:"top_k"`

	// Internal fields populated after load
	Aliases []string `yaml:"-"`
}

func Load(path string) (*Config, error) {
	cfg := &Config{}

	// 1. Determine config file path
	// If path is empty, try "config.yaml" in current directory and executable directory
	if path == "" {
		candidates := []string{"config.yaml"}

		// check executable dir
		exe, err := os.Executable()
		if err == nil {
			candidates = append(candidates, filepath.Join(filepath.Dir(exe), "config.yaml"))
		}

		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				path = c
				slog.Info("found configuration file", "path", path)
				break
			}
		}
	}

	// 2. Load from file if exists
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			if err := yaml.Unmarshal(data, cfg); err != nil {
				return nil, err
			}
		} else {
			slog.Warn("configured config file not found, skipping", "path", path)
		}
	}

	// 3. Override with Environment Variables (Priority: Env > File)
	if v := os.Getenv("RAG_BUCKET"); v != "" {
		cfg.Bucket = v
	}
	if v := os.Getenv("RAG_KB_ALIASES"); v != "" {
		cfg.Aliases = strings.Split(v, ",")
	} else if cfg.KBAlias != "" {
		cfg.Aliases = []string{cfg.KBAlias}
	}

	if v := os.Getenv("RAG_S3_ENDPOINT"); v != "" {
		cfg.S3Endpoint = v
	}
	if v := os.Getenv("RAG_S3_REGION"); v != "" {
		cfg.S3Region = v
	}
	if v := os.Getenv("MODEL"); v != "" {
		cfg.ModelName = v
	}
	if v := os.Getenv("RAG_MODELS_DIR"); v != "" {
		cfg.ModelsDir = v
	}
	if v := os.Getenv("RAG_INDEX_DIR"); v != "" {
		cfg.IndexDir = v
	}

	if v := os.Getenv("RAG_RRF_K"); v != "" {
		if i, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.RRFK = i
		}
	}
	if v := os.Getenv("RAG_LIMIT_VECTOR"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.LimitVector = i
		}
	}
	if v := os.Getenv("RAG_LIMIT_FTS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.LimitFTS = i
		}
	}
	if v := os.Getenv("RAG_TOP_K"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.TopK = i
		}
	}

	// 4. Set Defaults if still empty
	if cfg.RRFK == 0 {
		cfg.RRFK = 60
	}
	if cfg.LimitVector == 0 {
		cfg.LimitVector = 20
	}
	if cfg.LimitFTS == 0 {
		cfg.LimitFTS = 20
	}
	if cfg.TopK == 0 {
		cfg.TopK = 5
	}
	if cfg.S3Region == "" {
		cfg.S3Region = "us-east-1"
	}

	return cfg, nil
}
