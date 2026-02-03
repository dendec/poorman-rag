package search

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/dendec/poorman-rag/internal/domain/mcp"
	"github.com/dendec/poorman-rag/internal/embedding"
	"github.com/dendec/poorman-rag/internal/infrastructure/search"
	"github.com/dendec/poorman-rag/internal/utils"
)

// SearchService defines the interface for search operations
type SearchService interface {
	Search(ctx context.Context, query mcp.SearchQuery) ([]mcp.SearchResult, error)
	GetKnowledgeBases() []mcp.KnowledgeBase
}

// Service implements the SearchService interface and manages multiple knowledge bases
type Service struct {
	// Configuration
	opts     utils.S3Options
	indexDir string
	embedder embedding.Embedder
	dim      int

	// RRF Settings
	rrfK        float64
	limitVector int
	limitFTS    int
	topK        int

	// Knowledge base management
	aliases map[string]bool
	repositories map[string]*search.LanceDBRepository
	mu      sync.RWMutex

	// Default search type
	searchType string
}

// NewService creates a new search service
func NewService(opts utils.S3Options, indexDir string, aliases []string, dim int, embedder embedding.Embedder, searchType string) *Service {
	allowed := make(map[string]bool)
	for _, a := range aliases {
		allowed[a] = true
	}

	return &Service{
		opts:       opts,
		indexDir:   indexDir,
		aliases:    allowed,
		repositories:    make(map[string]*search.LanceDBRepository),
		dim:        dim,
		embedder:   embedder,
		searchType: searchType,
		// Default RRF settings
		rrfK:        search.DefaultRRFK,
		limitVector: search.DefaultLimitVector,
		limitFTS:    search.DefaultLimitFTS,
		topK:        search.DefaultTopK,
	}
}

// SetRRFConfig updates the RRF configuration
func (s *Service) SetRRFConfig(k float64, vector, fts, topK int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rrfK = k
	s.limitVector = vector
	s.limitFTS = fts
	s.topK = topK
}

// GetRepository returns a ready-to-use LanceDBRepository for the specified knowledge base.
func (s *Service) GetRepository(alias string) (*search.LanceDBRepository, error) {
	s.mu.RLock()
	if !s.aliases[alias] {
		s.mu.RUnlock()
		return nil, fmt.Errorf("unknown knowledge base alias: %s", alias)
	}
	repo, exists := s.repositories[alias]
	s.mu.RUnlock()
	if exists {
		return repo, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if repo, exists = s.repositories[alias]; exists {
		return repo, nil
	}

	slog.Info("lazy loading knowledge base", "kb", alias)

	var uri string
	tableName := alias // Default table name matches alias

	if s.indexDir != "" {
		// Local directory: e.g., /path/to/indices/jokes_lancedb
		uri = filepath.Join(s.indexDir, fmt.Sprintf("%s_lancedb", alias))
		if _, err := os.Stat(uri); err != nil {
			// Fallback to just alias dir
			uri = filepath.Join(s.indexDir, alias)
		}
	} else {
		// S3 URI: lancedb can connect to s3://bucket/prefix
		uri = fmt.Sprintf("s3://%s/rag/index/%s", s.opts.Bucket, alias)
	}

	newRepo, err := search.NewLanceDBRepository(uri, tableName, s.dim, s.embedder,
		search.WithQueryPrefix("query: "),
		search.WithRRFConfig(s.rrfK, s.limitVector, s.limitFTS, s.topK),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to init repository %s: %w", alias, err)
	}

	s.repositories[alias] = newRepo
	slog.Info("knowledge base ready", "kb", alias)
	return newRepo, nil
}

// ListAliases returns a list of available knowledge bases
func (s *Service) ListAliases() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	keys := make([]string, 0, len(s.aliases))
	for k := range s.aliases {
		keys = append(keys, k)
	}
	return keys
}

// Search performs a search using the underlying search repository
func (s *Service) Search(ctx context.Context, query mcp.SearchQuery) ([]mcp.SearchResult, error) {
	// Get the search repository for the knowledge base
	repository, err := s.GetRepository(query.KB)
	if err != nil {
		return nil, err
	}

	// Perform the search based on the configured search type
	var results []search.Result
	switch s.searchType {
	case mcp.SearchVector:
		results, err = repository.VectorSearch(ctx, query.Query)
	case mcp.SearchFTS:
		results, err = repository.FTSSearch(ctx, query.Query)
	case mcp.SearchHybrid:
		results, err = repository.HybridSearch(ctx, query.Query)
	default:
		results, err = repository.HybridSearch(ctx, query.Query) // default to hybrid
	}

	if err != nil {
		return nil, err
	}

	// Convert results to domain models
	domainResults := make([]mcp.SearchResult, len(results))
	for i, result := range results {
		domainResults[i] = mcp.SearchResult{
			Text:   result.Text,
			Score:  result.Score,
			Source: result.Source.String(),
		}
	}

	return domainResults, nil
}

// GetKnowledgeBases returns the list of available knowledge bases
func (s *Service) GetKnowledgeBases() []mcp.KnowledgeBase {
	aliases := s.ListAliases()
	kbs := make([]mcp.KnowledgeBase, len(aliases))

	for i, alias := range aliases {
		kbs[i] = mcp.KnowledgeBase{
			Name:        alias,
			Description: alias, // For now, using alias as description
		}
	}

	return kbs
}

// Close closes all repositories
func (s *Service) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range s.repositories {
		r.Close()
	}
}
