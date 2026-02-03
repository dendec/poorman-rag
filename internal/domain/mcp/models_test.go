package mcp

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRequest(t *testing.T) {
	t.Run("RequestSerialization", func(t *testing.T) {
		req := Request{
			JSONRPC: "2.0",
			Method:  "test_method",
			Params:  json.RawMessage(`{"param": "value"}`),
			ID:      1,
		}

		data, err := json.Marshal(req)
		assert.NoError(t, err)
		assert.Contains(t, string(data), "test_method")
	})
}

func TestResponse(t *testing.T) {
	t.Run("ResponseWithError", func(t *testing.T) {
		resp := Response{
			JSONRPC: "2.0",
			ID:      1,
			Error: &Error{
				Code:    -32603,
				Message: "test error",
			},
		}

		data, err := json.Marshal(resp)
		assert.NoError(t, err)
		assert.Contains(t, string(data), "test error")
	})

	t.Run("ResponseWithResult", func(t *testing.T) {
		resp := Response{
			JSONRPC: "2.0",
			ID:      1,
			Result:  map[string]interface{}{"success": true},
		}

		data, err := json.Marshal(resp)
		assert.NoError(t, err)
		assert.Contains(t, string(data), "success")
	})
}

func TestSearchQuery(t *testing.T) {
	t.Run("SearchQueryCreation", func(t *testing.T) {
		query := SearchQuery{
			Query: "test query",
			KB:    "test_kb",
		}

		assert.Equal(t, "test query", query.Query)
		assert.Equal(t, "test_kb", query.KB)
	})
}

func TestMCPServerInterface(t *testing.T) {
	t.Run("InterfaceImplementationCheck", func(t *testing.T) {
		// Verify that we can define the interface
		var server MCPServer
		assert.Nil(t, server)
	})
}
