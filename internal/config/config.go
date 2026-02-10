package config

import (
	"log/slog"
	"os"
	"path/filepath"

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

	// 3. Set defaults for aliases if KBAlias is set but Aliases is empty
	if cfg.KBAlias != "" && len(cfg.Aliases) == 0 {
		cfg.Aliases = []string{cfg.KBAlias}
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
