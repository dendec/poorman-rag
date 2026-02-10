package domain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEmbedding(t *testing.T) {
	t.Run("EmbeddingCanBeConvertedToSlice", func(t *testing.T) {
		embedding := Embedding{0.1, 0.2, 0.3}
		assert.Len(t, embedding, 3)
		assert.Equal(t, float32(0.1), embedding[0])
		assert.Equal(t, float32(0.3), embedding[2])
	})

	t.Run("EmptyEmbedding", func(t *testing.T) {
		embedding := Embedding{}
		assert.Empty(t, embedding)
	})
}

func TestModel(t *testing.T) {
	t.Run("ModelCreation", func(t *testing.T) {
		model := Model{
			Name:    "test-model",
			Type:    "onnx",
			Dim:     384,
			Pooling: "mean",
			Path:    "/path/to/model",
		}

		assert.Equal(t, "test-model", model.Name)
		assert.Equal(t, "onnx", model.Type)
		assert.Equal(t, 384, model.Dim)
		assert.Equal(t, "mean", model.Pooling)
		assert.Equal(t, "/path/to/model", model.Path)
	})
}
