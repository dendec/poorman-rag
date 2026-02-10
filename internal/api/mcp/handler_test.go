package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/dendec/poorman-rag/internal/domain"
	mcp_service "github.com/dendec/poorman-rag/internal/services/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockSearchService is a mock implementation of the search service
type MockSearchService struct {
	mock.Mock
}

func (m *MockSearchService) Search(ctx context.Context, query domain.SearchQuery) ([]domain.SearchResult, error) {
	args := m.Called(ctx, query)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]domain.SearchResult), args.Error(1)
}

func (m *MockSearchService) GetKnowledgeBases() []domain.KnowledgeBase {
	args := m.Called()
	return args.Get(0).([]domain.KnowledgeBase)
}

// createTestHandler creates a handler with a mock search service for testing
func createTestHandler(mockSearch *MockSearchService) *Handler {
	service := mcp_service.NewService(mockSearch)
	return NewHandler(service)
}

func TestHandler_Process_Initialize(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(`{"jsonrpc":"2.0","method":"initialize","id":1}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "2.0", response.JSONRPC)
	assert.Equal(t, float64(1), response.ID) // JSON unmarshals numbers as float64
	assert.NotNil(t, response.Result)

	result, ok := response.Result.(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "2024-11-05", result["protocolVersion"])
	assert.Contains(t, result, "capabilities")
	assert.Contains(t, result, "serverInfo")
}

func TestHandler_Process_NotificationInitialized(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(`{"jsonrpc":"2.0","method":"notifications/initialized"}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.Nil(t, response) // Notifications return nil
}

func TestHandler_Process_Ping(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(`{"jsonrpc":"2.0","method":"ping","id":2}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "2.0", response.JSONRPC)
	assert.Equal(t, float64(2), response.ID)
	assert.NotNil(t, response.Result)
}

func TestHandler_Process_ToolsList(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	// Mock the GetKnowledgeBases call
	mockSearch.On("GetKnowledgeBases").Return([]domain.KnowledgeBase{
		{Name: "jokes", Description: "Jokes knowledge base"},
		{Name: "docs", Description: "Documentation"},
	})

	reqBody := []byte(`{"jsonrpc":"2.0","method":"tools/list","id":3}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "2.0", response.JSONRPC)
	assert.Equal(t, float64(3), response.ID)

	result, ok := response.Result.(map[string]interface{})
	assert.True(t, ok)
	assert.Contains(t, result, "tools")

	mockSearch.AssertExpectations(t)
}

func TestHandler_Process_ToolsCall_Success(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	// Mock successful search
	mockSearch.On("Search", mock.Anything, domain.SearchQuery{
		Query: "funny joke",
		KB:    "jokes",
	}).Return([]domain.SearchResult{
		{Text: "Why did the chicken cross the road?", Score: 0.95, Source: "joke1.txt"},
		{Text: "To get to the other side!", Score: 0.85, Source: "joke2.txt"},
	}, nil)

	params := domain.ToolCallParams{
		Name: "search_jokes",
		Arguments: map[string]string{
			"query": "funny joke",
		},
	}
	paramsJSON, _ := json.Marshal(params)
	reqBody := []byte(`{"jsonrpc":"2.0","method":"tools/call","params":` + string(paramsJSON) + `,"id":4}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "2.0", response.JSONRPC)
	assert.Equal(t, float64(4), response.ID)
	assert.Nil(t, response.Error)

	result, ok := response.Result.(*domain.ToolCallResult)
	assert.True(t, ok)
	assert.False(t, result.IsError)
	assert.Len(t, result.Content, 2)
	assert.Equal(t, "text", result.Content[0].Type)
	assert.Contains(t, result.Content[0].Text, "chicken")

	mockSearch.AssertExpectations(t)
}

func TestHandler_Process_ToolsCall_SearchError(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	// Mock search error
	mockSearch.On("Search", mock.Anything, domain.SearchQuery{
		Query: "test",
		KB:    "unknown",
	}).Return([]domain.SearchResult(nil), errors.New("knowledge base not found"))

	params := domain.ToolCallParams{
		Name: "search_unknown",
		Arguments: map[string]string{
			"query": "test",
		},
	}
	paramsJSON, _ := json.Marshal(params)
	reqBody := []byte(`{"jsonrpc":"2.0","method":"tools/call","params":` + string(paramsJSON) + `,"id":5}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)

	result, ok := response.Result.(*domain.ToolCallResult)
	assert.True(t, ok)
	assert.True(t, result.IsError)
	assert.Len(t, result.Content, 1)
	assert.Contains(t, result.Content[0].Text, "Error:")

	mockSearch.AssertExpectations(t)
}

func TestHandler_Process_ToolsCall_MissingQuery(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	params := domain.ToolCallParams{
		Name:      "search_jokes",
		Arguments: map[string]string{}, // Missing query
	}
	paramsJSON, _ := json.Marshal(params)
	reqBody := []byte(`{"jsonrpc":"2.0","method":"tools/call","params":` + string(paramsJSON) + `,"id":6}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotNil(t, response.Error)
	assert.Equal(t, -32603, response.Error.Code)
	assert.Contains(t, response.Error.Message, "query is required")
}

func TestHandler_Process_ToolsCall_InvalidToolName(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	params := domain.ToolCallParams{
		Name: "invalid_tool", // Doesn't start with "search_"
		Arguments: map[string]string{
			"query": "test",
		},
	}
	paramsJSON, _ := json.Marshal(params)
	reqBody := []byte(`{"jsonrpc":"2.0","method":"tools/call","params":` + string(paramsJSON) + `,"id":7}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotNil(t, response.Error)
	assert.Equal(t, -32603, response.Error.Code)
	assert.Contains(t, response.Error.Message, "unknown tool")
}

func TestHandler_Process_MethodNotFound(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(`{"jsonrpc":"2.0","method":"unknown_method","id":8}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.NotNil(t, response.Error)
	assert.Equal(t, -32601, response.Error.Code)
	assert.Contains(t, response.Error.Message, "Method not found")
}

func TestHandler_Process_InvalidJSON(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(`{invalid json}`)

	response, err := handler.Process(context.Background(), reqBody)

	assert.Nil(t, response)
	assert.Error(t, err)
}

func TestHandler_Process_EmptyBody(t *testing.T) {
	mockSearch := new(MockSearchService)
	handler := createTestHandler(mockSearch)

	reqBody := []byte(``)

	response, err := handler.Process(context.Background(), reqBody)

	assert.Nil(t, response)
	assert.Error(t, err)
}

func TestNewHandler(t *testing.T) {
	mockSearch := new(MockSearchService)
	service := mcp_service.NewService(mockSearch)
	handler := NewHandler(service)

	assert.NotNil(t, handler)
	assert.NotNil(t, handler.service)
}
