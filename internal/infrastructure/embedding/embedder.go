package embedding

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/dendec/poorman-rag/internal/domain"
	"github.com/dendec/poorman-rag/internal/embedding/onnx"
)

// ONNXEmbedder implements the domain.Embedder interface using ONNX runtime
type ONNXEmbedder struct {
	embedder *onnx.GenericEmbedder
	config   onnx.Config
}

// NewONNXEmbedder creates a new ONNX embedder
func NewONNXEmbedder(libPath string, config onnx.Config) (*ONNXEmbedder, error) {
	embedder, err := onnx.NewGenericEmbedder(libPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX embedder: %w", err)
	}

	return &ONNXEmbedder{
		embedder: embedder,
		config:   config,
	}, nil
}

// Embed implements the domain.Embedder interface
func (oe *ONNXEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		// Perform the embedding operation
		result, err := oe.embedder.Embed(context.Background(), text)
		if err != nil {
			return nil, fmt.Errorf("ONNX embedding failed: %w", err)
		}
		return result, nil
	}
}

// GetModelInfo returns information about the underlying model
func (oe *ONNXEmbedder) GetModelInfo() domain.Model {
	return domain.Model{
		Name:    filepath.Base(oe.config.ModelPath), // Using filename as model name
		Type:    "onnx",
		Dim:     int(oe.config.Dimensions),
		Pooling: string(oe.config.Pooling),
		Path:    filepath.Dir(oe.config.ModelPath), // directory containing the model files
	}
}

// Close releases resources held by the embedder
func (oe *ONNXEmbedder) Close() error {
	if oe.embedder != nil {
		return oe.embedder.Close()
	}
	return nil
}
