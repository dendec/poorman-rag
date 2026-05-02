package search

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/dendec/poorman-rag/internal/config"
	"github.com/dendec/poorman-rag/internal/embedding/onnx"
	"github.com/dendec/poorman-rag/internal/utils"
)

// Manager manage multiple search engines (Knowledge Bases)
type Manager struct {
	opts     utils.S3Options
	indexDir string
	modelDir string

	// Map of settings per alias
	kbSettings map[string]config.KBSettings

	// Cache of initialized engines
	engines map[string]*Engine
	mu      sync.RWMutex
}

func NewManager(opts utils.S3Options, indexDir, modelDir string, kbSettings map[string]config.KBSettings) *Manager {
	return &Manager{
		opts:       opts,
		indexDir:   indexDir,
		modelDir:   modelDir,
		kbSettings: kbSettings,
		engines:    make(map[string]*Engine),
	}
}

// ListAliases returns a list of available knowledge bases
func (m *Manager) ListAliases() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	keys := make([]string, 0, len(m.kbSettings))
	for k := range m.kbSettings {
		keys = append(keys, k)
	}
	return keys
}

// GetEngine returns a ready-to-use Engine.
func (m *Manager) GetEngine(alias string) (*Engine, error) {
	m.mu.RLock()
	settings, exists := m.kbSettings[alias]
	if !exists {
		m.mu.RUnlock()
		return nil, fmt.Errorf("unknown knowledge base alias: %s", alias)
	}
	eng, exists := m.engines[alias]
	m.mu.RUnlock()
	if exists {
		return eng, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if eng, exists = m.engines[alias]; exists {
		return eng, nil
	}

	slog.Info("lazy loading knowledge base", "kb", alias)
	eng, err := m.loadEngine(alias, settings)
	if err != nil {
		return nil, err
	}

	m.engines[alias] = eng
	slog.Info("knowledge base ready", "kb", alias)
	return eng, nil
}

func (m *Manager) loadEngine(alias string, settings config.KBSettings) (*Engine, error) {
	// 1. Prepare keys and paths
	modelSlug := m.getModelSlug(settings.ModelName)
	modelKeys := []string{
		"rag/models/" + modelSlug + "/model_quantized.onnx",
		"rag/models/" + modelSlug + "/tokenizer.json",
		"rag/models/" + modelSlug + "/model_config.json",
	}

	baseS3Path := fmt.Sprintf("rag/index/%s", alias)
	dbKey := baseS3Path + "/dataset.sqlite.zst"
	idxKey := baseS3Path + "/vectors.usearch.zst"

	localModelDir := m.findModelDir(modelSlug, settings.ModelDir)
	localDB := filepath.Join("/tmp", fmt.Sprintf("%s_dataset.sqlite", alias))
	localIdx := filepath.Join("/tmp", fmt.Sprintf("%s_vectors.usearch", alias))

	// 2. Determine what's missing
	modelPaths, keysToDownload, downloadIdx := m.resolveModelFiles(modelKeys, localModelDir)

	actualDB, actualIdx, foundLocal := m.checkLocalIndex(alias, settings.IndexDir)
	if foundLocal {
		localDB = actualDB
		localIdx = actualIdx
	} else {
		keysToDownload = append(keysToDownload, dbKey, idxKey)
		downloadIdx = append(downloadIdx, len(modelKeys), len(modelKeys)+1)
	}

	// 3. Download if needed
	if len(keysToDownload) > 0 {
		if err := m.downloadAndMap(alias, keysToDownload, downloadIdx, modelPaths, localDB, localIdx); err != nil {
			return nil, err
		}
	}

	// 4. Initialize everything
	return m.initEngine(alias, settings, modelPaths, localDB, localIdx)
}

func (m *Manager) getModelSlug(modelName string) string {
	slug := modelName
	if parts := strings.Split(slug, "/"); len(parts) > 1 {
		slug = parts[len(parts)-1]
	}
	return slug
}

func (m *Manager) resolveModelFiles(keys []string, localDir string) ([]string, []string, []int) {
	paths := make([]string, len(keys))
	var toDownload []string
	var indices []int

	for i, key := range keys {
		path := filepath.Join(localDir, filepath.Base(key))
		if _, err := os.Stat(path); err == nil {
			paths[i] = path
		} else {
			toDownload = append(toDownload, key)
			indices = append(indices, i)
		}
	}
	return paths, toDownload, indices
}

func (m *Manager) downloadAndMap(alias string, keys []string, indices []int, modelPaths []string, localDB, localIdx string) error {
	files, err := utils.EnsureDownload(m.opts, keys...)
	if err != nil {
		return fmt.Errorf("failed to download artifacts for %s: %w", alias, err)
	}

	modelKeysCount := len(modelPaths)
	for i, filePath := range files {
		origIdx := indices[i]
		if origIdx < modelKeysCount {
			modelPaths[origIdx] = filePath
		} else if origIdx == modelKeysCount {
			if err := os.Rename(filePath, localDB); err != nil {
				return fmt.Errorf("rename db: %w", err)
			}
		} else if origIdx == modelKeysCount+1 {
			if err := os.Rename(filePath, localIdx); err != nil {
				return fmt.Errorf("rename idx: %w", err)
			}
		}
	}
	return nil
}

func (m *Manager) initEngine(alias string, settings config.KBSettings, modelPaths []string, localDB, localIdx string) (*Engine, error) {
	onnxCfg, err := onnx.LoadConfig(modelPaths[2])
	if err != nil {
		return nil, err
	}
	onnxCfg.ModelPath = modelPaths[0]
	onnxCfg.TokenizerPath = modelPaths[1]
	if settings.PoolingMode != "" {
		onnxCfg.Pooling = onnx.PoolingType(settings.PoolingMode)
	}
	if settings.Dimensions > 0 {
		onnxCfg.Dimensions = settings.Dimensions
	}

	libPath := filepath.Join("lib", "libonnxruntime.so")
	embedder, err := onnx.NewGenericEmbedder(libPath, onnxCfg)
	if err != nil {
		return nil, fmt.Errorf("init embedder %s: %w", alias, err)
	}

	return NewEngine(localDB, localIdx, onnxCfg.Dimensions, embedder,
		WithQueryPrefix(settings.QueryPrefix),
		WithRRFConfig(settings.RRFK, settings.LimitVector, settings.LimitFTS, settings.TopK),
	)
}

func (m *Manager) findModelDir(slug string, settingsModelDir string) string {
	var candidates []string
	if settingsModelDir != "" {
		candidates = append(candidates, filepath.Join(settingsModelDir, slug), settingsModelDir)
	}
	if m.modelDir != "" {
		candidates = append(candidates, filepath.Join(m.modelDir, slug), m.modelDir)
	}
	candidates = append(candidates,
		filepath.Join("models", slug),
		filepath.Join("indexer", "models", slug),
		"models",
		"indexer/models",
	)

	for _, dir := range candidates {
		if dir == "" {
			continue
		}
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir
		}
	}
	return "/tmp"
}

func (m *Manager) checkLocalIndex(alias string, settingsIndexDir string) (string, string, bool) {
	idxDir := m.indexDir
	if settingsIndexDir != "" {
		idxDir = settingsIndexDir
	}

	if idxDir == "" {
		// Also check in /tmp from previous runs
		db := filepath.Join("/tmp", fmt.Sprintf("%s_dataset.sqlite", alias))
		idx := filepath.Join("/tmp", fmt.Sprintf("%s_vectors.usearch", alias))
		if _, err := os.Stat(db); err == nil {
			if _, err := os.Stat(idx); err == nil {
				return db, idx, true
			}
		}
		return "", "", false
	}

	// Multi-index check: idxDir/alias/dataset.sqlite
	candidateDB := filepath.Join(idxDir, alias, "dataset.sqlite")
	candidateIdx := filepath.Join(idxDir, alias, "vectors.usearch")

	// Single-index check: idxDir/dataset.sqlite
	if _, err := os.Stat(candidateDB); err != nil {
		candidateDB = filepath.Join(idxDir, "dataset.sqlite")
		candidateIdx = filepath.Join(idxDir, "vectors.usearch")
	}

	if _, err := os.Stat(candidateDB); err == nil {
		if _, err := os.Stat(candidateIdx); err == nil {
			return candidateDB, candidateIdx, true
		}
	}
	return "", "", false
}

func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.engines {
		e.Close()
	}
}
