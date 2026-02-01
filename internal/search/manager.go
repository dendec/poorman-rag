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
	dim      int

	// RRF Settings to pass to new engines
	rrfK        float64
	limitVector int
	limitFTS    int
	topK        int

	// List of allowed aliases (from config)
	aliases map[string]bool

	// Cache of initialized engines
	engines map[string]*Engine
	mu      sync.RWMutex
}

func NewManager(opts utils.S3Options, indexDir string, aliases []string, dim int, embedder embedding.Embedder) *Manager {
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
	m.limitVector = vector
	m.limitFTS = fts
	m.topK = topK
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

	var uri string
	tableName := alias // Default table name matches alias

	if m.indexDir != "" {
		// Local directory: e.g., /path/to/indices/jokes_lancedb
		uri = filepath.Join(m.indexDir, fmt.Sprintf("%s_lancedb", alias))
		if _, err := os.Stat(uri); err != nil {
			// Fallback to just alias dir
			uri = filepath.Join(m.indexDir, alias)
		}
	} else {
		// S3 URI: lancedb can connect to s3://bucket/prefix
		uri = fmt.Sprintf("s3://%s/rag/index/%s", m.opts.Bucket, alias)
	}

	newEng, err := NewEngine(uri, tableName, m.dim, m.embedder,
		WithQueryPrefix("query: "),
		WithRRFConfig(m.rrfK, m.limitVector, m.limitFTS, m.topK),
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
