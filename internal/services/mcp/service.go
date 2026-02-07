package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/dendec/poorman-rag/internal/domain"
	search_service "github.com/dendec/poorman-rag/internal/services/search"
)

// Service provides MCP functionality
type Service struct {
	searchService search_service.SearchService
}

// NewService creates a new MCP service
func NewService(searchService search_service.SearchService) *Service {
	return &Service{
		searchService: searchService,
	}
}

// Process handles an MCP request
func (s *Service) Process(ctx context.Context, reqBody []byte) (*domain.Response, error) {
	var req domain.Request
	if err := json.Unmarshal(reqBody, &req); err != nil {
		return nil, err
	}

	resp := &domain.Response{
		JSONRPC: "2.0",
		ID:      req.ID,
	}

	switch req.Method {
	case "initialize":
		resp.Result = s.initialize()
	case "notifications/initialized":
		// Client acknowledging initialization. No response needed, but we must not error.
		return nil, nil
	case "ping":
		// Standard liveness check
		resp.Result = map[string]interface{}{}
	case "tools/list":
		resp.Result = s.listTools()
	case "tools/call":
		result, err := s.callTool(ctx, req.Params)
		if err != nil {
			resp.Error = &domain.Error{Code: -32603, Message: err.Error()}
		} else {
			resp.Result = result
		}
	default:
		// MCP requires ignoring unknown methods or returning an error
		resp.Error = &domain.Error{Code: -32601, Message: fmt.Sprintf("Method not found: %s", req.Method)}
	}

	return resp, nil
}

// initialize returns the initialization response
func (s *Service) initialize() map[string]interface{} {
	return map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities": map[string]interface{}{
			"tools": map[string]interface{}{},
		},
		"serverInfo": map[string]interface{}{
			"name":    "poorman-rag",
			"version": "0.1.0",
		},
	}
}

// listTools returns the list of available tools
func (s *Service) listTools() map[string]interface{} {
	kbs := s.searchService.GetKnowledgeBases()
	tools := make([]domain.Tool, 0, len(kbs))

	for _, kb := range kbs {
		toolName := fmt.Sprintf("search_%s", kb.Name)
		description := fmt.Sprintf("Search in the '%s' knowledge base.", kb.Description)

		tools = append(tools, domain.Tool{
			Name:        toolName,
			Description: description,
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Search query text",
					},
				},
				"required": []string{"query"},
			},
		})
	}

	return map[string]interface{}{
		"tools": tools,
	}
}

// callTool handles a tool call (internal method)
func (s *Service) callTool(ctx context.Context, paramsRaw json.RawMessage) (*domain.ToolCallResult, error) {
	var params domain.ToolCallParams
	if err := json.Unmarshal(paramsRaw, &params); err != nil {
		return nil, fmt.Errorf("invalid params: %v", err)
	}

	// Determine knowledge base from tool name
	// search_jokes -> jokes
	if !strings.HasPrefix(params.Name, "search_") {
		return nil, fmt.Errorf("unknown tool: %s", params.Name)
	}

	kbName := strings.TrimPrefix(params.Name, "search_")

	query := params.Arguments["query"]
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}

	slog.Info("incoming MCP search", "kb", kbName, "query", query)

	// Perform search
	searchQuery := domain.SearchQuery{
		Query: query,
		KB:    kbName,
	}

	results, err := s.searchService.Search(ctx, searchQuery)
	if err != nil {
		return &domain.ToolCallResult{
			Content: []domain.ToolContent{{Type: "text", Text: fmt.Sprintf("Error: %v", err)}},
			IsError: true,
		}, nil
	}

	// Log statistics
	slog.Info("search finished", "kb", kbName, "total", len(results))

	content := make([]domain.ToolContent, 0, len(results))
	for _, res := range results {
		content = append(content, domain.ToolContent{Type: "text", Text: res.Text})
	}

	return &domain.ToolCallResult{Content: content}, nil
}
