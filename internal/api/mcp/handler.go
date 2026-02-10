package mcp

import (
	"context"

	"github.com/dendec/poorman-rag/internal/domain"
	mcp_service "github.com/dendec/poorman-rag/internal/services/mcp"
)

// Handler adapts the MCP service to the HTTP API
type Handler struct {
	service *mcp_service.Service
}

// NewHandler creates a new MCP API handler
func NewHandler(service *mcp_service.Service) *Handler {
	return &Handler{
		service: service,
	}
}

// Process handles an MCP request
func (a *Handler) Process(ctx context.Context, reqBody []byte) (*domain.Response, error) {
	// The service expects raw bytes and returns a response
	serviceResp, err := a.service.Process(ctx, reqBody)
	if err != nil {
		return nil, err
	}

	// If service returns nil (for notifications), return nil
	if serviceResp == nil {
		return nil, nil
	}

	return serviceResp, nil
}