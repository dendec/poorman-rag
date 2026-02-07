package embedding

import (
	"context"
	"errors"
	"fmt"

	"github.com/dendec/poorman-rag/internal/domain"
)

// Service provides embedding functionality
type Service struct {
	embedder domain.Embedder
	model    string
}

// NewService creates a new embedding service
func NewService(embedder domain.Embedder, model string) *Service {
	return &Service{
		embedder: embedder,
		model:    model,
	}
}

// ComputeEmbedding computes embedding for a single text
func (s *Service) ComputeEmbedding(ctx context.Context, text string) (domain.Embedding, error) {
	if text == "" {
		return nil, errors.New("text cannot be empty")
	}

	vec, err := s.embedder.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to compute embedding: %w", err)
	}

	return domain.Embedding(vec), nil
}

// ComputeEmbeddings computes embeddings for multiple texts
func (s *Service) ComputeEmbeddings(ctx context.Context, texts []string) ([]domain.Embedding, error) {
	if len(texts) == 0 {
		return nil, errors.New("texts cannot be empty")
	}

	embeddings := make([]domain.Embedding, len(texts))
	for i, text := range texts {
		embedding, err := s.ComputeEmbedding(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to compute embedding for text at index %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// GetModel returns the model name used by this service
func (s *Service) GetModel() string {
	return s.model
}

// ValidateEmbedding validates that the embedding has the expected dimensions
func (s *Service) ValidateEmbedding(embedding domain.Embedding, expectedDim int) error {
	if len(embedding) != expectedDim {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", expectedDim, len(embedding))
	}
	return nil
}
