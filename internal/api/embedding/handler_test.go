package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dendec/poorman-rag/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockEmbeddingService is a mock implementation that matches the embedding.Service struct methods
type MockEmbeddingService struct {
	mock.Mock
}

func (m *MockEmbeddingService) ComputeEmbedding(ctx context.Context, text string) (domain.Embedding, error) {
	args := m.Called(ctx, text)
	return args.Get(0).(domain.Embedding), args.Error(1)
}

func (m *MockEmbeddingService) ComputeEmbeddings(ctx context.Context, texts []string) ([]domain.Embedding, error) {
	args := m.Called(ctx, texts)
	return args.Get(0).([]domain.Embedding), args.Error(1)
}

func (m *MockEmbeddingService) GetModel() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockEmbeddingService) ValidateEmbedding(embedding domain.Embedding, expectedDim int) error {
	args := m.Called(embedding, expectedDim)
	return args.Error(0)
}

func TestHandler_HandleOpenAIRequest_StringInput(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	expectedEmbedding := domain.Embedding{0.1, 0.2, 0.3}
	mockService.On("ComputeEmbeddings", mock.Anything, []string{"hello world"}).Return([]domain.Embedding{expectedEmbedding}, nil)
	mockService.On("GetModel").Return("test-model")

	req := domain.EmbeddingRequest{
		Input: "hello world",
		Model: "test-model",
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	assert.Equal(t, "test-model", response.Model)
	assert.Len(t, response.Data, 1)
	assert.Equal(t, "embedding", response.Data[0].Object)
	assert.Equal(t, 0, response.Data[0].Index)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, response.Data[0].Embedding)
	
	mockService.AssertExpectations(t)
}

func TestHandler_HandleOpenAIRequest_ArrayInput(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	embeddings := []domain.Embedding{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	mockService.On("ComputeEmbeddings", mock.Anything, []string{"hello", "world"}).Return(embeddings, nil)
	mockService.On("GetModel").Return("test-model")

	req := domain.EmbeddingRequest{
		Input: []interface{}{"hello", "world"},
		Model: "test-model",
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	assert.Equal(t, "test-model", response.Model)
	assert.Len(t, response.Data, 2)
	assert.Equal(t, "embedding", response.Data[0].Object)
	assert.Equal(t, 0, response.Data[0].Index)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, response.Data[0].Embedding)
	assert.Equal(t, 1, response.Data[1].Index)
	assert.Equal(t, []float32{0.4, 0.5, 0.6}, response.Data[1].Embedding)
	
	mockService.AssertExpectations(t)
}

func TestHandler_HandleOpenAIRequest_NumberArray(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	embeddings := []domain.Embedding{
		{0.1, 0.2, 0.3},
	}
	mockService.On("ComputeEmbeddings", mock.Anything, []string{"123", "456"}).Return(embeddings, nil)
	mockService.On("GetModel").Return("test-model")

	req := domain.EmbeddingRequest{
		Input: []interface{}{float64(123), float64(456)},
		Model: "test-model",
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	assert.Equal(t, "test-model", response.Model)
	assert.Len(t, response.Data, 1)
	assert.Equal(t, "embedding", response.Data[0].Object)
	assert.Equal(t, 0, response.Data[0].Index)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, response.Data[0].Embedding)
	
	mockService.AssertExpectations(t)
}

func TestHandler_HandleOpenAIRequest_EmptyStringInput(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	expectedEmbedding := domain.Embedding{0.1, 0.2, 0.3}
	mockService.On("ComputeEmbeddings", mock.Anything, []string{""}).Return([]domain.Embedding{expectedEmbedding}, nil)
	mockService.On("GetModel").Return("test-model")

	req := domain.EmbeddingRequest{
		Input: "",
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	
	mockService.AssertExpectations(t)
}

func TestHandler_HandleOpenAIRequest_EmptyArray(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	req := domain.EmbeddingRequest{
		Input: []interface{}{},
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.Nil(t, response)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no input provided")
	
	mockService.AssertNotCalled(t, "ComputeEmbeddings")
}

func TestHandler_HandleOpenAIRequest_InvalidInputType(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	req := domain.EmbeddingRequest{
		Input: 123, // Invalid type
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.Nil(t, response)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported input type")
	
	mockService.AssertNotCalled(t, "ComputeEmbeddings")
}

func TestHandler_HandleOpenAIRequest_ServiceError(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	mockService.On("ComputeEmbeddings", mock.Anything, []string{"error text"}).Return([]domain.Embedding(nil), errors.New("service error"))

	req := domain.EmbeddingRequest{
		Input: "error text",
	}

	response, err := handler.HandleOpenAIRequest(context.Background(), req)

	assert.Nil(t, response)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to compute embeddings")
	assert.Contains(t, err.Error(), "service error")
	
	mockService.AssertExpectations(t)
}

func TestHandler_ComputeEmbedding_Success(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	expectedEmbedding := domain.Embedding{0.1, 0.2, 0.3}
	mockService.On("ComputeEmbedding", mock.Anything, "hello").Return(expectedEmbedding, nil)

	result, err := handler.ComputeEmbedding(context.Background(), "hello")

	assert.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, result)
	
	mockService.AssertExpectations(t)
}

func TestHandler_ComputeEmbedding_Error(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	mockService.On("ComputeEmbedding", mock.Anything, "error text").Return(domain.Embedding(nil), errors.New("compute error"))

	result, err := handler.ComputeEmbedding(context.Background(), "error text")

	assert.Nil(t, result)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "compute error")
	
	mockService.AssertExpectations(t)
}

func TestHandler_HTTPHandler_ValidRequest(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	// Prepare mock expectations
	expectedEmbedding := domain.Embedding{0.1, 0.2, 0.3}
	mockService.On("ComputeEmbeddings", mock.Anything, []string{"hello world"}).Return([]domain.Embedding{expectedEmbedding}, nil)
	mockService.On("GetModel").Return("test-model")

	// Create HTTP request with JSON body
	jsonData := `{"input": "hello world", "model": "test-model"}`
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(jsonData))
	req.Header.Set("Content-Type", "application/json")
	
	// Create response recorder
	rr := httptest.NewRecorder()
	
	// Call the handler
	handler.HTTPHandler(rr, req)
	
	// Check the status code
	assert.Equal(t, http.StatusOK, rr.Code)
	
	// Check the response body
	var response domain.EmbeddingResponse
	err := json.Unmarshal(rr.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	assert.Equal(t, "test-model", response.Model)
	assert.Len(t, response.Data, 1)
	assert.Equal(t, "embedding", response.Data[0].Object)
	assert.Equal(t, 0, response.Data[0].Index)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, response.Data[0].Embedding)
	
	mockService.AssertExpectations(t)
}

func TestHandler_HTTPHandler_InvalidMethod(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	
	handler.HTTPHandler(rr, req)
	
	assert.Equal(t, http.StatusMethodNotAllowed, rr.Code)
	
	mockService.AssertNotCalled(t, "ComputeEmbeddings")
}

func TestHandler_HTTPHandler_InvalidJSON(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	invalidJSON := `{invalid json}`
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(invalidJSON))
	req.Header.Set("Content-Type", "application/json")
	
	rr := httptest.NewRecorder()
	
	handler.HTTPHandler(rr, req)
	
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	
	mockService.AssertNotCalled(t, "ComputeEmbeddings")
}

func TestHandler_HTTPHandler_ServiceError(t *testing.T) {
	mockService := new(MockEmbeddingService)
	handler := NewHandler(mockService)

	// Setup mock to return an error
	mockService.On("ComputeEmbeddings", mock.Anything, []string{"error text"}).Return([]domain.Embedding(nil), errors.New("service error"))

	jsonData := `{"input": "error text", "model": "test-model"}`
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(jsonData))
	req.Header.Set("Content-Type", "application/json")
	
	rr := httptest.NewRecorder()
	
	handler.HTTPHandler(rr, req)
	
	assert.Equal(t, http.StatusInternalServerError, rr.Code)
	
	mockService.AssertExpectations(t)
}