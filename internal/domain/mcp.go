package domain

import (
	"context"
	"encoding/json"
)

// Request represents an MCP request
type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
	ID      interface{}     `json:"id"`
}

// Response represents an MCP response
type Response struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *Error      `json:"error,omitempty"`
	ID      interface{} `json:"id"`
}

// Error represents an MCP error
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Tool represents an MCP tool
type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"inputSchema"`
}

// ToolCallParams represents parameters for calling a tool
type ToolCallParams struct {
	Name      string            `json:"name"`
	Arguments map[string]string `json:"arguments"`
}

// ToolCallResult represents the result of calling a tool
type ToolCallResult struct {
	Content []ToolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ToolContent represents content in a tool result
type ToolContent struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

// KnowledgeBase represents a knowledge base
type KnowledgeBase struct {
	Name        string
	Description string
}

// SearchQuery represents a search query
type SearchQuery struct {
	Query string
	KB    string
}

// SearchResult represents a search result
type SearchResult struct {
	Text   string
	Score  float64
	Source string
}

// MCPServer defines the interface for an MCP server
type MCPServer interface {
	Process(ctx context.Context, reqBody []byte) (*Response, error)
	Initialize(ctx context.Context) *Response
	ListTools(ctx context.Context) *Response
	CallTool(ctx context.Context, params ToolCallParams) *Response
}

const (
	SearchVector = "vector"
	SearchFTS    = "fts"
	SearchHybrid = "hybrid"
)
