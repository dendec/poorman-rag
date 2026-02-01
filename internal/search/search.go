package search

import (
	"context"
	"fmt"
	"log/slog"
	"sort"

	"github.com/dendec/poorman-rag/internal/embedding"
	"github.com/lancedb/lancedb-go/pkg/contracts"
	"github.com/lancedb/lancedb-go/pkg/lancedb"
)

const (
	DefaultRRFK        = 60.0
	DefaultLimitVector = 20
	DefaultLimitFTS    = 20
	DefaultTopK        = 10
)

type Source int

const (
	Vector Source = 1 << iota
	FTS
)

type Result struct {
	ID     int64   `json:"id"`
	Text   string  `json:"text"`
	Score  float64 `json:"score"`
	Source Source  `json:"source"`
}

type Engine struct {
	db          contracts.IConnection
	table       contracts.ITable
	embedder    embedding.Embedder
	queryPrefix string

	// RRF Parameters
	rrfK        float64
	limitVector int
	limitFTS    int
	topK        int
}

type NewOption func(*Engine)

func NewEngine(uri, tableName string, dim int, embedder embedding.Embedder, opts ...NewOption) (*Engine, error) {
	ctx := context.Background()

	// 1. LanceDB Connection
	db, err := lancedb.Connect(ctx, uri, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to lancedb: %w", err)
	}

	// 2. Open Table
	table, err := db.OpenTable(ctx, tableName)
	if err != nil {
		return nil, fmt.Errorf("failed to open table %s: %w", tableName, err)
	}

	// 3. Engine creation
	e := &Engine{
		db:          db,
		table:       table,
		embedder:    embedder,
		rrfK:        DefaultRRFK,
		limitVector: DefaultLimitVector,
		limitFTS:    DefaultLimitFTS,
		topK:        DefaultTopK,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e, nil
}

func WithQueryPrefix(prefix string) NewOption {
	return func(e *Engine) { e.queryPrefix = prefix }
}

func WithRRFConfig(k float64, vector, fts, topK int) NewOption {
	return func(e *Engine) {
		if k > 0 {
			e.rrfK = k
		}
		if vector > 0 {
			e.limitVector = vector
		}
		if fts > 0 {
			e.limitFTS = fts
		}
		if topK > 0 {
			e.topK = topK
		}
	}
}

func (e *Engine) Close() {
	// LanceDB connections usually don't need explicit close in Go if using shared context,
	// but Engine should be clean.
}

func (e *Engine) prepareQuery(q string) string {
	if e.queryPrefix == "" {
		return q
	}
	return e.queryPrefix + q
}

func (e *Engine) HybridSearch(ctx context.Context, query string) ([]Result, error) {
	// 1. Vector Search
	queryVec, err := e.embedder.Embed(ctx, e.prepareQuery(query))
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	// LanceDB Hybrid search (not yet fully optimized in Go SDK similarly to Python).
	// We will perform two searches and fuse them manually to keep RRF control.

	// A. Vector Search
	vecResults, err := e.table.VectorSearch(ctx, "vector", queryVec, e.limitVector)
	if err != nil {
		slog.Warn("vector search failed", "error", err)
	}

	// B. FTS Search
	ftsResults, err := e.table.FullTextSearch(ctx, "text", query)
	if err != nil {
		slog.Warn("fts search failed", "error", err)
	}

	// 3. RRF Fusion
	scores := make(map[int64]float64)
	sources := make(map[int64]Source)
	textMap := make(map[int64]string)

	e.processMapResults(vecResults, Vector, scores, sources, textMap)
	e.processMapResults(ftsResults, FTS, scores, sources, textMap)

	// 4. Sorting and Top-K
	return e.mapToResults(scores, sources, textMap), nil
}

func (e *Engine) processMapResults(results []map[string]interface{}, source Source, scores map[int64]float64, sources map[int64]Source, textMap map[int64]string) {
	for rank, row := range results {
		idVal, ok := row["id"]
		if !ok {
			continue
		}

		var id int64
		switch v := idVal.(type) {
		case int64:
			id = v
		case float64:
			id = int64(v)
		case int32:
			id = int64(v)
		default:
			continue
		}

		textVal, ok := row["text"]
		if !ok {
			continue
		}
		text, _ := textVal.(string)

		scores[id] += 1.0 / (e.rrfK + float64(rank+1))
		sources[id] |= source
		textMap[id] = text
	}
}

func (e *Engine) VectorSearch(ctx context.Context, query string) ([]Result, error) {
	queryVec, err := e.embedder.Embed(ctx, e.prepareQuery(query))
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}
	results, err := e.table.VectorSearch(ctx, "vector", queryVec, e.topK)
	if err != nil {
		return nil, fmt.Errorf("vector search error: %w", err)
	}

	scores := make(map[int64]float64)
	sources := make(map[int64]Source)
	textMap := make(map[int64]string)
	e.processMapResults(results, Vector, scores, sources, textMap)

	return e.mapToResults(scores, sources, textMap), nil
}

func (e *Engine) FTSSearch(ctx context.Context, query string) ([]Result, error) {
	results, err := e.table.FullTextSearch(ctx, "text", query)
	if err != nil {
		return nil, fmt.Errorf("fts search error: %w", err)
	}
	// Note: FullTextSearch might not support Limit in the same way,
	// but the doc says FullTextSearch(ctx, column, query).
	// If it needs limit we might need another method.

	scores := make(map[int64]float64)
	sources := make(map[int64]Source)
	textMap := make(map[int64]string)
	e.processMapResults(results, FTS, scores, sources, textMap)

	return e.mapToResults(scores, sources, textMap), nil
}

func (e *Engine) mapToResults(scores map[int64]float64, sources map[int64]Source, textMap map[int64]string) []Result {
	type item struct {
		id    int64
		score float64
	}
	var sorted []item
	for id, score := range scores {
		sorted = append(sorted, item{id, score})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})

	results := make([]Result, 0, len(sorted))
	for _, it := range sorted {
		if txt, ok := textMap[it.id]; ok {
			results = append(results, Result{
				ID:     it.id,
				Text:   txt,
				Score:  it.score,
				Source: sources[it.id],
			})
		}
	}
	return results
}
