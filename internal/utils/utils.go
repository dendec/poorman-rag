package utils

import (
	"fmt"
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

// FileExists checks if a file exists and is not a directory
func FileExists(path string) error {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return err
	}
	if err != nil {
		return err
	}
	if info.IsDir() {
		return fmt.Errorf("path is a directory: %s", path)
	}
	if info.Size() == 0 {
		return fmt.Errorf("file is empty: %s", path)
	}
	return nil
}

// DirExists checks if a directory exists
func DirExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return err == nil && info.IsDir()
}
