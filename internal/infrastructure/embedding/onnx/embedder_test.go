package onnx

import (
	"testing"

	"github.com/dendec/poorman-rag/internal/domain/embedding"
)

func TestONNXEmbedder(t *testing.T) {
	t.Run("InterfaceImplementation", func(t *testing.T) {
		// Verify that ONNXEmbedder implements the embedding.Embedder interface
		var _ embedding.Embedder = (*ONNXEmbedder)(nil)
	})
}