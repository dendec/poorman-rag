package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/dendec/poorman-rag/internal/search"
)

const (
	SearchVector = "vector"
	SearchFTS    = "fts"
	SearchHybrid = "hybrid"
)

var searchFunc = map[string]func(*search.Engine, context.Context, string) ([]search.Result, error){
	SearchVector: (*search.Engine).VectorSearch,
	SearchFTS:    (*search.Engine).FTSSearch,
	SearchHybrid: (*search.Engine).HybridSearch,
}

type Handler struct {
	manager    *search.Manager // Reference to the manager
	searchType string
}

func NewHandler(manager *search.Manager, searchType string) *Handler {
	if _, ok := searchFunc[searchType]; !ok {
		// Fallback to hybrid if invalid
		searchType = SearchHybrid
	}
	return &Handler{
		manager:    manager,
		searchType: searchType,
	}
}

func (h *Handler) Process(ctx context.Context, reqBody []byte) (*JSONRPCResponse, error) {
	var req JSONRPCRequest
	if err := json.Unmarshal(reqBody, &req); err != nil {
		return nil, err
	}

	resp := &JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      req.ID,
	}

	switch req.Method {
	case "initialize":
		resp.Result = map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]interface{}{
				"tools": map[string]interface{}{},
			},
			"serverInfo": map[string]interface{}{
				"name":    "poorman-rag",
				"version": "0.1.0",
			},
		}
	case "notifications/initialized":
		// Client acknowledging initialization. No response needed, but we must not error.
		return nil, nil
	case "ping":
		// Standard liveness check
		resp.Result = map[string]interface{}{}
	case "tools/list":
		resp.Result = h.handleListTools()
	case "tools/call":
		res, err := h.handleCallTool(ctx, req.Params)
		if err != nil {
			resp.Error = &RPCError{Code: -32603, Message: err.Error()}
		} else {
			resp.Result = res
		}
	default:
		// MCP requires ignoring unknown methods or returning an error
		resp.Error = &RPCError{Code: -32601, Message: fmt.Sprintf("Method not found: %s", req.Method)}
	}

	return resp, nil
}

func (h *Handler) handleListTools() map[string]interface{} {
	tools := []Tool{}

	// Dynamically create tools for each alias
	// jokes -> search_jokes
	// prompts -> search_prompts
	for _, alias := range h.manager.ListAliases() {
		toolName := fmt.Sprintf("search_%s", alias)
		description := fmt.Sprintf("Search in the '%s' knowledge base.", alias)

		tools = append(tools, Tool{
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

func (h *Handler) handleCallTool(ctx context.Context, paramsRaw json.RawMessage) (*CallToolResult, error) {
	var params CallParams
	if err := json.Unmarshal(paramsRaw, &params); err != nil {
		return nil, fmt.Errorf("invalid params: %v", err)
	}

	// 1. Determine alias from tool name
	// search_jokes -> jokes
	if !strings.HasPrefix(params.Name, "search_") {
		return nil, fmt.Errorf("unknown tool: %s", params.Name)
	}

	alias := strings.TrimPrefix(params.Name, "search_")

	// 2. Get (lazy) engine for this alias
	engine, err := h.manager.GetEngine(alias)
	if err != nil {
		return nil, fmt.Errorf("failed to load KB '%s': %v", alias, err)
	}

	query := params.Arguments["query"]
	if query == "" {
		return nil, errors.New("query is required")
	}

	slog.Info("incoming MCP search", "kb", alias, "query", query, "type", h.searchType)

	// 3. Choose search function
	fn := searchFunc[h.searchType]

	// 4. Search
	results, err := fn(engine, ctx, query)
	if err != nil {
		return &CallToolResult{
			Content: []ToolContent{{Type: "text", Text: fmt.Sprintf("Error: %v", err)}},
			IsError: true,
		}, nil
	}

	// Log statistics
	var ftsBits, vectorBits uint8
	for _, r := range results {
		if r.Source&search.FTS == search.FTS {
			ftsBits++
		}
		if r.Source&search.Vector == search.Vector {
			vectorBits++
		}
	}
	slog.Info("search finished", "kb", alias, "total", len(results), "fts", ftsBits, "vector", vectorBits)
	content := make([]ToolContent, 0, len(results))
	for _, res := range results {
		content = append(content, ToolContent{Type: "text", Text: res.Text})
	}
	return &CallToolResult{Content: content}, nil
}
