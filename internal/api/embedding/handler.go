package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/dendec/poorman-rag/internal/domain"
)

// Handler adapts the embedding service to the HTTP API
type Handler struct {
	service domain.EmbeddingService
}

// NewHandler creates a new embedding API handler
func NewHandler(service domain.EmbeddingService) *Handler {
	return &Handler{
		service: service,
	}
}

// HandleOpenAIRequest handles OpenAI-compatible embedding requests
func (a *Handler) HandleOpenAIRequest(ctx context.Context, req domain.EmbeddingRequest) (*domain.EmbeddingResponse, error) {
	var texts []string

	// Parse input - can be string, []string, or []int
	switch v := req.Input.(type) {
	case string:
		texts = []string{v}
	case []interface{}:
		for _, item := range v {
			switch val := item.(type) {
			case string:
				texts = append(texts, val)
			case float64: // JSON numbers are parsed as float64
				texts = append(texts, fmt.Sprintf("%.0f", val)) // Convert to string representation
			default:
				return nil, fmt.Errorf("unsupported input type in array: %T", val)
			}
		}
	default:
		return nil, fmt.Errorf("unsupported input type: %T", v)
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("no input provided")
	}

	// Compute embeddings for all texts using the service layer
	embeddingsDomain, err := a.service.ComputeEmbeddings(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("failed to compute embeddings: %w", err)
	}

	// Convert domain embeddings to the API format
	embeddings := make([]domain.EmbeddingResult, len(embeddingsDomain))
	for i, embedding := range embeddingsDomain {
		embeddings[i] = domain.EmbeddingResult{
			Object:    "embedding",
			Index:     i,
			Embedding: []float32(embedding),
		}
	}

	// Calculate usage stats (simple approximation)
	totalTokens := 0
	for _, text := range texts {
		totalTokens += len(text) // Rough token count
	}

	response := &domain.EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  a.service.GetModel(),
		Usage: domain.Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	return response, nil
}

// ComputeEmbedding computes embeddings for a single text
func (a *Handler) ComputeEmbedding(ctx context.Context, text string) ([]float32, error) {
	embedding, err := a.service.ComputeEmbedding(ctx, text)
	if err != nil {
		return nil, err
	}

	// Convert domain embedding to []float32
	return []float32(embedding), nil
}

// HTTPHandler returns an HTTP handler for the embedding API
func (a *Handler) HTTPHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		slog.Error("failed to read request body", "error", err)
		http.Error(w, "Failed to read request", http.StatusBadRequest)
		return
	}

	var embeddingReq domain.EmbeddingRequest
	if err := json.Unmarshal(body, &embeddingReq); err != nil {
		slog.Error("invalid request body", "error", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	response, err := a.HandleOpenAIRequest(ctx, embeddingReq)
	if err != nil {
		slog.Error("embedding request processing error", "error", err)
		http.Error(w, "Internal Error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
