package embedding

import (
	"context"
	"errors"
	"testing"

	"github.com/dendec/poorman-rag/internal/domain/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockEmbedder is a mock implementation of the Embedder interface
type MockEmbedder struct {
	mock.Mock
}

func (m *MockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	args := m.Called(ctx, text)
	return args.Get(0).([]float32), args.Error(1)
}

func TestService(t *testing.T) {
	t.Run("NewService", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		assert.NotNil(t, service)
		assert.Equal(t, "test-model", service.GetModel())
	})

	t.Run("ComputeEmbedding_Success", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		expectedEmbedding := []float32{0.1, 0.2, 0.3}
		mockEmbedder.On("Embed", mock.Anything, "hello world").Return(expectedEmbedding, nil)
		
		ctx := context.Background()
		result, err := service.ComputeEmbedding(ctx, "hello world")
		
		assert.NoError(t, err)
		assert.Equal(t, embedding.Embedding(expectedEmbedding), result)
		mockEmbedder.AssertExpectations(t)
	})

	t.Run("ComputeEmbedding_EmptyText", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		ctx := context.Background()
		result, err := service.ComputeEmbedding(ctx, "")
		
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "text cannot be empty")
	})

	t.Run("ComputeEmbedding_ErrorFromEmbedder", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		mockEmbedder.On("Embed", mock.Anything, "hello world").Return([]float32(nil), errors.New("embedder error"))
		
		ctx := context.Background()
		result, err := service.ComputeEmbedding(ctx, "hello world")
		
		assert.Error(t, err)
		assert.Nil(t, result)
		assert.Contains(t, err.Error(), "failed to compute embedding")
		mockEmbedder.AssertExpectations(t)
	})

	t.Run("ComputeEmbeddings_Success", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		mockEmbedder.On("Embed", mock.Anything, "hello").Return([]float32{0.1, 0.2}, nil)
		mockEmbedder.On("Embed", mock.Anything, "world").Return([]float32{0.3, 0.4}, nil)
		
		ctx := context.Background()
		results, err := service.ComputeEmbeddings(ctx, []string{"hello", "world"})
		
		assert.NoError(t, err)
		assert.Len(t, results, 2)
		assert.Equal(t, embedding.Embedding([]float32{0.1, 0.2}), results[0])
		assert.Equal(t, embedding.Embedding([]float32{0.3, 0.4}), results[1])
		mockEmbedder.AssertExpectations(t)
	})

	t.Run("ComputeEmbeddings_EmptyInputs", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		ctx := context.Background()
		results, err := service.ComputeEmbeddings(ctx, []string{})
		
		assert.Error(t, err)
		assert.Nil(t, results)
		assert.Contains(t, err.Error(), "texts cannot be empty")
	})

	t.Run("ComputeEmbeddings_ErrorInMiddle", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		mockEmbedder.On("Embed", mock.Anything, "hello").Return([]float32{0.1, 0.2}, nil)
		mockEmbedder.On("Embed", mock.Anything, "error").Return([]float32(nil), errors.New("embedder error"))
		
		ctx := context.Background()
		results, err := service.ComputeEmbeddings(ctx, []string{"hello", "error"})
		
		assert.Error(t, err)
		assert.Nil(t, results)
		assert.Contains(t, err.Error(), "failed to compute embedding for text at index 1")
		mockEmbedder.AssertExpectations(t)
	})

	t.Run("ValidateEmbedding_Valid", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		embedding := embedding.Embedding{0.1, 0.2, 0.3}
		err := service.ValidateEmbedding(embedding, 3)
		
		assert.NoError(t, err)
	})

	t.Run("ValidateEmbedding_InvalidDimension", func(t *testing.T) {
		mockEmbedder := &MockEmbedder{}
		service := NewService(mockEmbedder, "test-model")
		
		embedding := embedding.Embedding{0.1, 0.2, 0.3}
		err := service.ValidateEmbedding(embedding, 4)
		
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "embedding dimension mismatch: expected 4, got 3")
	})
}