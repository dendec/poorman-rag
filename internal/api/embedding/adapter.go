package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	services_embedding "github.com/dendec/poorman-rag/internal/services/embedding"
)

// EmbeddingRequest represents the OpenAI-compatible embedding request
type EmbeddingRequest struct {
	Input any    `json:"input"`           // Can be string, []string, or []int
	Model string `json:"model,omitempty"` // Model identifier
	User  string `json:"user,omitempty"`  // Optional user identifier
}

// EmbeddingResponse represents the OpenAI-compatible embedding response
type EmbeddingResponse struct {
	Object string         `json:"object"`
	Data   []EmbeddingObj `json:"data"`
	Model  string         `json:"model"`
	Usage  Usage          `json:"usage"`
}

// EmbeddingObj represents a single embedding result
type EmbeddingObj struct {
	Object    string      `json:"object"`
	Index     int         `json:"index"`
	Embedding []float32   `json:"embedding"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// APIAdapter adapts the embedding service to the HTTP API
type APIAdapter struct {
	service *services_embedding.Service
}

// NewAPIAdapter creates a new API adapter
func NewAPIAdapter(service *services_embedding.Service) *APIAdapter {
	return &APIAdapter{
		service: service,
	}
}

// HandleOpenAIRequest handles OpenAI-compatible embedding requests
func (a *APIAdapter) HandleOpenAIRequest(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
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
	embeddings := make([]EmbeddingObj, len(embeddingsDomain))
	for i, embedding := range embeddingsDomain {
		embeddings[i] = EmbeddingObj{
			Object:    "embedding",
			Index:     i,
			Embedding: []float32(embedding),
		}
	}
	
	// Calculate usage stats (simple approximation)
	totalTokens := 0
	for _, text := range texts {
		totalTokens += len(strings.Fields(text)) // Rough token count
	}
	
	response := &EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  a.service.GetModel(),
		Usage: Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}
	
	return response, nil
}

// ComputeEmbedding computes embeddings for a single text
func (a *APIAdapter) ComputeEmbedding(ctx context.Context, text string) ([]float32, error) {
	embedding, err := a.service.ComputeEmbedding(ctx, text)
	if err != nil {
		return nil, err
	}
	
	// Convert domain embedding to []float32
	return []float32(embedding), nil
}

// HTTPHandler returns an HTTP handler for the embedding API
func (a *APIAdapter) HTTPHandler(w http.ResponseWriter, r *http.Request) {
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

	var embeddingReq EmbeddingRequest
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