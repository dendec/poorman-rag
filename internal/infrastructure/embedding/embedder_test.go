package embedding

import (
	"testing"

	"github.com/dendec/poorman-rag/internal/domain"
)

func TestONNXEmbedder(t *testing.T) {
	t.Run("InterfaceImplementation", func(t *testing.T) {
		// Verify that ONNXEmbedder implements the domain.Embedder interface
		var _ domain.Embedder = (*ONNXEmbedder)(nil)
	})
}
