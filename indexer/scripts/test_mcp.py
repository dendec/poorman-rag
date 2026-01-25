import requests
import json
import argparse
import sys
import os

def test_mcp_search(url, domain, query, top_k=1):
    """
    Sends MCP 'tools/call' request (search)
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": f"search_{domain}",
            "arguments": {
                "query": query
            }
        },
        "id": 1
    }
    
    print(f"🔍 [{domain.upper()}] Query: '{query}'")
    print(f"🌐 URL: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            print(f"❌ JSON-RPC Error: {data['error']}")
            return
        
        result = data.get("result", {})
        if result is None:
            print("⚠️ Response result is None. Is the tool name correct?")
            return

        content_list = result.get("content", [])

        if not content_list:
            print("⚠️ Response content list is empty.")
            return

        for i, item in enumerate(content_list[:top_k]):
            text_result = item.get("text", "")
            print(f"\n--- Result #{i+1} ---")
            # Print first 800 characters nicely
            print(text_result[:800] + ("..." if len(text_result) > 800 else ""))
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")

def test_mcp_list(url):
    """
    Sends MCP 'tools/list' request
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    
    print(f"📋 Requesting tools list from {url}...")
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        tools = response.json().get("result", {}).get("tools", [])
        for t in tools:
            print(f" - {t['name']}: {t['description']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test poorman-rag MCP Lambda")
    parser.add_argument("--url", "-u", help="Lambda Function URL (required or set MCP_LAMBDA_URL env var)")
    parser.add_argument("--domain", "-d", required=True, help="Knowledge base alias to search (e.g., 'docs')")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--list", "-l", action="store_true", help="List available tools")
    
    args = parser.parse_args()
    
    target_url = args.url or os.getenv("MCP_LAMBDA_URL")
    if not target_url:
        print("❌ Error: Lambda URL is not specified. Use --url or set MCP_LAMBDA_URL environment variable.")
        sys.exit(1)

    if args.list:
        test_mcp_list(target_url)
    else:
        if not args.query:
            print("❌ Error: --query is required when not in --list mode.")
            sys.exit(1)
        test_mcp_search(target_url, args.domain, args.query)