package search

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"sort"
	"strings"

	"github.com/dendec/poorman-rag/internal/embedding"

	usearch "github.com/unum-cloud/usearch/golang"
	_ "modernc.org/sqlite" // Pure Go SQLite driver
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
	ID       int64                  `json:"id"`
	Text     string                 `json:"text"`
	Metadata map[string]interface{} `json:"metadata"`
	Score    float64                `json:"score"`
	Source   Source                 `json:"source"`
}

type Engine struct {
	db          *sql.DB
	index       *usearch.Index
	embedder    embedding.Embedder
	queryPrefix string
	hasFTS      bool

	// RRF Parameters
	rrfK        float64
	limitVector uint
	limitFTS    int
	topK        uint
}

type NewOption func(*Engine)

func NewEngine(dbPath, indexPath string, dim uint, embedder embedding.Embedder, opts ...NewOption) (*Engine, error) {
	// 1. SQLite (Read-Only)
	dsn := fmt.Sprintf("file:%s?mode=ro", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open db: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("db ping failed: %w", err)
	}

	// 2. USEARCH Index
	conf := usearch.DefaultConfig(dim)
	index, err := usearch.NewIndex(conf)
	if err != nil {
		slog.Error("failed to create usearch index", "error", err)
		return nil, fmt.Errorf("usearch init failed: %w", err)
	}
	err = index.View(indexPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load index: %w", err)
	}

	// 3. Engine creation
	e := &Engine{
		db:          db,
		index:       index,
		embedder:    embedder,
		rrfK:        DefaultRRFK,
		limitVector: uint(DefaultLimitVector),
		limitFTS:    DefaultLimitFTS,
		topK:        uint(DefaultTopK),
	}

	for _, opt := range opts {
		opt(e)
	}

	// 4. Check for FTS table
	var name string
	err = db.QueryRow("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_fts'").Scan(&name)
	if err == nil {
		e.hasFTS = true
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
			e.limitVector = uint(vector)
		}
		if fts > 0 {
			e.limitFTS = fts
		}
		if topK > 0 {
			e.topK = uint(topK)
		}
	}
}

func (e *Engine) Close() {
	e.db.Close()
	e.index.Destroy()
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
	vectorIDs, _, err := e.index.Search(queryVec, e.limitVector)
	if err != nil {
		return nil, fmt.Errorf("vector search error: %w", err)
	}

	// 2. FTS Search
	ftsIDs, err := e.searchFTS(query, e.limitFTS)
	if err != nil {
		slog.Warn("FTS search failed", "error", err, "query", query)
		ftsIDs = []int64{}
	}

	// 3. RRF Fusion
	scores := make(map[int64]float64)
	sources := make(map[int64]Source)

	for rank, id := range vectorIDs {
		id64 := int64(id)
		scores[id64] += 1.0 / (e.rrfK + float64(rank+1))
		sources[id64] |= Vector
	}

	for rank, id := range ftsIDs {
		scores[id] += 1.0 / (e.rrfK + float64(rank+1))
		sources[id] |= FTS
	}

	// 4. Sorting and Top-K
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

	maxTopK := int(e.topK)
	if len(sorted) > maxTopK {
		sorted = sorted[:maxTopK]
	}

	if len(sorted) == 0 {
		return []Result{}, nil
	}

	// 5. Data Hydration
	ids := make([]interface{}, len(sorted))
	for i, it := range sorted {
		ids[i] = it.id
	}

	placeholders := strings.Repeat("?,", len(ids)-1) + "?"
	querySQL := fmt.Sprintf("SELECT id, text, metadata FROM dataset WHERE id IN (%s)", placeholders)

	rows, err := e.db.Query(querySQL, ids...)
	if err != nil {
		return nil, fmt.Errorf("db fetch error: %w", err)
	}
	defer rows.Close()

	type dataRecord struct {
		text     string
		metadata map[string]interface{}
	}
	textMap := make(map[int64]dataRecord)
	for rows.Next() {
		var id int64
		var text string
		var metaStr sql.NullString
		if err := rows.Scan(&id, &text, &metaStr); err != nil {
			continue
		}
		var meta map[string]interface{}
		if metaStr.Valid && metaStr.String != "" {
			json.Unmarshal([]byte(metaStr.String), &meta)
		}
		textMap[id] = dataRecord{text, meta}
	}

	results := make([]Result, 0, len(sorted))
	for _, it := range sorted {
		if rec, ok := textMap[it.id]; ok {
			results = append(results, Result{
				ID:       it.id,
				Text:     rec.text,
				Metadata: rec.metadata,
				Score:    it.score,
				Source:   sources[it.id],
			})
		}
	}

	return results, nil
}

var ftsSanitizer = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)

func (e *Engine) searchFTS(query string, limit int) ([]int64, error) {
	if !e.hasFTS {
		return []int64{}, nil
	}
	safeQuery := ftsSanitizer.ReplaceAllString(query, " ")
	tokens := strings.Fields(safeQuery)
	if len(tokens) == 0 {
		return []int64{}, nil
	}

	ftsQuery := strings.Join(tokens, " OR ")
	querySQL := `SELECT rowid FROM dataset_fts WHERE dataset_fts MATCH ? ORDER BY rank LIMIT ?`

	rows, err := e.db.Query(querySQL, ftsQuery, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			continue
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func (e *Engine) VectorSearch(ctx context.Context, query string) ([]Result, error) {
	queryVec, err := e.embedder.Embed(ctx, e.prepareQuery(query))
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}
	vectorIDs, _, err := e.index.Search(queryVec, e.topK)
	if err != nil {
		return nil, fmt.Errorf("vector search error: %w", err)
	}

	if len(vectorIDs) == 0 {
		return []Result{}, nil
	}

	ids := make([]interface{}, len(vectorIDs))
	for i, id := range vectorIDs {
		ids[i] = int64(id)
	}

	placeholders := strings.Repeat("?,", len(ids)-1) + "?"
	querySQL := fmt.Sprintf("SELECT id, text, metadata FROM dataset WHERE id IN (%s)", placeholders)

	rows, err := e.db.Query(querySQL, ids...)
	if err != nil {
		return nil, fmt.Errorf("db fetch error: %w", err)
	}
	defer rows.Close()

	type dataRecord struct {
		text     string
		metadata map[string]interface{}
	}
	textMap := make(map[int64]dataRecord)
	for rows.Next() {
		var id int64
		var text string
		var metaStr sql.NullString
		if err := rows.Scan(&id, &text, &metaStr); err != nil {
			continue
		}
		var meta map[string]interface{}
		if metaStr.Valid && metaStr.String != "" {
			json.Unmarshal([]byte(metaStr.String), &meta)
		}
		textMap[id] = dataRecord{text, meta}
	}

	results := make([]Result, 0, len(vectorIDs))
	for i, id := range vectorIDs {
		if rec, ok := textMap[int64(id)]; ok {
			results = append(results, Result{
				ID:       int64(id),
				Text:     rec.text,
				Metadata: rec.metadata,
				Score:    1.0 / (float64(i) + 1.0),
				Source:   Vector,
			})
		}
	}
	return results, nil
}

func (e *Engine) FTSSearch(ctx context.Context, query string) ([]Result, error) {
	ids, err := e.searchFTS(query, int(e.topK))
	if err != nil {
		return nil, fmt.Errorf("fts search error: %w", err)
	}

	if len(ids) == 0 {
		return []Result{}, nil
	}

	placeholders := strings.Repeat("?,", len(ids)-1) + "?"
	querySQL := fmt.Sprintf("SELECT id, text, metadata FROM dataset WHERE id IN (%s)", placeholders)
	idsAny := make([]interface{}, len(ids))
	for i, id := range ids {
		idsAny[i] = id
	}
	rows, err := e.db.Query(querySQL, idsAny...)
	if err != nil {
		return nil, fmt.Errorf("db fetch error: %w", err)
	}
	defer rows.Close()

	type dataRecord struct {
		text     string
		metadata map[string]interface{}
	}
	textMap := make(map[int64]dataRecord)
	for rows.Next() {
		var id int64
		var text string
		var metaStr sql.NullString
		if err := rows.Scan(&id, &text, &metaStr); err != nil {
			continue
		}
		var meta map[string]interface{}
		if metaStr.Valid && metaStr.String != "" {
			json.Unmarshal([]byte(metaStr.String), &meta)
		}
		textMap[id] = dataRecord{text, meta}
	}

	results := make([]Result, 0, len(ids))
	for i, id := range ids {
		if rec, ok := textMap[id]; ok {
			results = append(results, Result{
				ID:       id,
				Text:     rec.text,
				Metadata: rec.metadata,
				Score:    1.0 / (float64(i) + 1.0),
				Source:   FTS,
			})
		}
	}
	return results, nil
}
