package embedding

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
