package config

import (
	"log/slog"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type KBSettings struct {
	ModelName   string  `yaml:"model_name"`
	QueryPrefix string  `yaml:"query_prefix"`
	PoolingMode string  `yaml:"pooling_mode"`
	RRFK        float64 `yaml:"rrf_k"`
	LimitVector int     `yaml:"limit_vector"`
	LimitFTS    int     `yaml:"limit_fts"`
	TopK        int     `yaml:"top_k"`
	Dimensions  uint    `yaml:"dimensions"`
	IndexDir    string  `yaml:"index_dir"` // Optional override
	ModelDir    string  `yaml:"model_dir"` // Optional override
}

type Config struct {
	// S3 / Cloud Storage (Global)
	Bucket      string `yaml:"s3_bucket"`
	S3Endpoint  string `yaml:"s3_endpoint"`
	S3Region    string `yaml:"s3_region"`
	S3AccessKey string `yaml:"s3_access_key"`
	S3SecretKey string `yaml:"s3_secret_key"`

	ModelDir string `yaml:"model_dir"`
	IndexDir string `yaml:"index_dir"`

	// Knowledge Bases
	KBs map[string]KBSettings `yaml:"kb"`

	// Internal fields
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

	// 3. Populate Aliases from Map
	for name := range cfg.KBs {
		cfg.Aliases = append(cfg.Aliases, name)
	}

	// 4. Override with Environment Variables (Global only for now)
	if v := os.Getenv("RAG_BUCKET"); v != "" {
		cfg.Bucket = v
	}
	if v := os.Getenv("RAG_MODEL_DIR"); v != "" {
		cfg.ModelDir = v
	}
	if v := os.Getenv("RAG_INDEX_DIR"); v != "" {
		cfg.IndexDir = v
	}

	// 5. Apply defaults and sanitize each KB
	for name, kb := range cfg.KBs {
		if kb.RRFK == 0 {
			kb.RRFK = 60
		}
		if kb.LimitVector == 0 {
			kb.LimitVector = 20
		}
		if kb.LimitFTS == 0 {
			kb.LimitFTS = 20
		}
		if kb.TopK == 0 {
			kb.TopK = 5
		}
		if kb.QueryPrefix == "" {
			kb.QueryPrefix = "query: "
		}
		cfg.KBs[name] = kb
	}

	if cfg.S3Region == "" {
		cfg.S3Region = "us-east-1"
	}

	return cfg, nil
}
