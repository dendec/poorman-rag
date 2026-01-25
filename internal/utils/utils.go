package utils

import (
	"log/slog"
	"os"
	"strconv"
)

func StringFromEnv(name string) string {
	val := os.Getenv(name)
	if val == "" {
		slog.Error("missing required environment variable", "variable", name)
		os.Exit(1)
	}
	return val
}

func StringFromEnvDefault(name, def string) string {
	val := os.Getenv(name)
	if val == "" {
		return def
	}
	return val
}

func IntFromEnv(name string) int {
	val := StringFromEnv(name)
	res, err := strconv.Atoi(val)
	if err != nil {
		slog.Error("invalid integer value for environment variable", "variable", name, "value", val)
		os.Exit(1)
	}
	return res
}

func IntFromEnvDefault(name string, def int) int {
	val := os.Getenv(name)
	if val == "" {
		return def
	}
	res, err := strconv.Atoi(val)
	if err != nil {
		slog.Warn("invalid integer value for environment variable, using default", "variable", name, "value", val, "default", def)
		return def
	}
	return res
}
