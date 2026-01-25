package search

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/dendec/poorman-rag/internal/embedding"
	"github.com/dendec/poorman-rag/internal/utils"
)

// Manager manage multiple search engines (Knowledge Bases)
type Manager struct {
	opts     utils.S3Options
	indexDir string // Local directory to search for indices first
	embedder embedding.Embedder
	dim      uint

	// RRF Settings to pass to new engines
	rrfK        float64
	limitVector uint
	limitFTS    int
	topK        uint

	// List of allowed aliases (from config)
	aliases map[string]bool

	// Cache of initialized engines
	engines map[string]*Engine
	mu      sync.RWMutex
}

func NewManager(opts utils.S3Options, indexDir string, aliases []string, dim uint, embedder embedding.Embedder) *Manager {
	allowed := make(map[string]bool)
	for _, a := range aliases {
		allowed[a] = true
	}

	return &Manager{
		opts:     opts,
		indexDir: indexDir,
		aliases:  allowed,
		engines:  make(map[string]*Engine),
		dim:      dim,
		embedder: embedder,
	}
}

func (m *Manager) SetRRFConfig(k float64, vector, fts, topK int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rrfK = k
	m.limitVector = uint(vector)
	m.limitFTS = fts
	m.topK = uint(topK)
}

// ListAliases returns a list of available knowledge bases
func (m *Manager) ListAliases() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	keys := make([]string, 0, len(m.aliases))
	for k := range m.aliases {
		keys = append(keys, k)
	}
	return keys
}

// GetEngine returns a ready-to-use Engine.
func (m *Manager) GetEngine(alias string) (*Engine, error) {
	m.mu.RLock()
	if !m.aliases[alias] {
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

	baseS3Path := fmt.Sprintf("rag/index/%s", alias)
	dbKey := baseS3Path + "/content.sqlite.zst"
	idxKey := baseS3Path + "/vectors.usearch"

	// Check local index directory if configured
	var localDB, localIdx string
	foundLocal := false

	if m.indexDir != "" {
		// If indexDir is configured, check for standard filenames (content.sqlite, vectors.usearch) within it.
		candidateDB := filepath.Join(m.indexDir, "content.sqlite")
		candidateIdx := filepath.Join(m.indexDir, "vectors.usearch")

		if _, err := os.Stat(candidateDB); err == nil {
			if _, err := os.Stat(candidateIdx); err == nil {
				localDB = candidateDB
				localIdx = candidateIdx
				foundLocal = true
				slog.Info("using local index files", "kb", alias, "db", localDB, "idx", localIdx)
			}
		}
	}

	if !foundLocal {
		localDB = filepath.Join("/tmp", fmt.Sprintf("%s_content.sqlite", alias))
		localIdx = filepath.Join("/tmp", fmt.Sprintf("%s_vectors.usearch", alias))

		files, err := utils.EnsureDownload(m.opts, dbKey, idxKey)
		if err != nil {
			return nil, fmt.Errorf("failed to download %s: %w", alias, err)
		}
		// ... rename logic ...
		if err := os.Rename(files[0], localDB); err != nil {
			return nil, fmt.Errorf("rename db tmp -> %s: %w", localDB, err)
		}
		if err := os.Rename(files[1], localIdx); err != nil {
			return nil, fmt.Errorf("rename idx tmp -> %s: %w", localIdx, err)
		}
	}

	newEng, err := NewEngine(localDB, localIdx, m.dim, m.embedder,
		WithQueryPrefix("query: "),
		WithRRFConfig(m.rrfK, int(m.limitVector), m.limitFTS, int(m.topK)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init engine %s: %w", alias, err)
	}

	m.engines[alias] = newEng
	slog.Info("knowledge base ready", "kb", alias)
	return newEng, nil
}

func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.engines {
		e.Close()
	}
}
