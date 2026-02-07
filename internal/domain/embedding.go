package domain

import "context"

// Embedding represents the embedding vector
type Embedding []float32

// Embedder defines the interface for embedding text
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// Model represents an embedding model
type Model struct {
	Name    string
	Type    string // e.g., "onnx", "transformers", etc.
	Dim     int
	Pooling string // e.g., "mean", "cls", etc.
	Path    string // Path to model files
}

// EmbeddingRequest represents the OpenAI-compatible embedding request
type EmbeddingRequest struct {
	Input any    `json:"input"`           // Can be string, []string, or []int
	Model string `json:"model,omitempty"` // Model identifier
	User  string `json:"user,omitempty"`  // Optional user identifier
}

// EmbeddingResponse represents the OpenAI-compatible embedding response
type EmbeddingResponse struct {
	Object string      `json:"object"`
	Data   []EmbeddingResult `json:"data"`
	Model  string      `json:"model"`
	Usage  Usage       `json:"usage"`
}

// EmbeddingResult represents a single embedding result in API response
type EmbeddingResult struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingService defines the interface for embedding service
type EmbeddingService interface {
	ComputeEmbedding(ctx context.Context, text string) (Embedding, error)
	ComputeEmbeddings(ctx context.Context, texts []string) ([]Embedding, error)
	GetModel() string
	ValidateEmbedding(embedding Embedding, expectedDim int) error
}