# MCP Configuration Guide

Once deployed, poorman-rag acts as an MCP server over HTTP.

## 1. Configure Claude Desktop
Add poorman-rag to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "poormanrag": {
      "command": "curl",
      "args": [
        "-s",
        "-X",
        "POST",
        "https://your-lambda-url/",
        "-H",
        "Content-Type: application/json",
        "-d",
        "{{REPLACE_ME}}"
      ]
    }
  }
}
```
*Note: Some clients might require a wrapper script if they don't support direct HTTP MCP servers yet. You can use a simple node/python bridge if needed.*

## 2. Available Tools
poorman-rag dynamically exposes tools based on your `RAG_KB_ALIASES`:
- `search_docs`: Search in the 'docs' knowledge base.
- `search_wiki`: Search in the 'wiki' knowledge base.
Each tool takes a single `query` argument.
