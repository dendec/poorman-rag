package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/dendec/poorman-rag/internal/app"
)

func main() {
	configPath := flag.String("config", "", "path to config.yaml file")
	flag.Parse()

	// Standard MCP Stdio servers often use JSON-RPC over stdin/stdout
	// For simplicity, we reuse our Handler.Process which takes a full request body

	app.InitSlog()
	ragApp, err := app.NewApp(*configPath)
	if err != nil {
		slog.Error("failed to initialize rag app", "error", err)
		os.Exit(1)
	}

	slog.Info("poorman-rag stdio server started. listening for JSON-RPC messages...")

	reader := bufio.NewReader(os.Stdin)
	for {
		// Read message from stdin (assuming one full JSON per line or similar protocol)
		// NOTE: MCP spec is more formal, but for CLI testing we can use a simpler loop
		line, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}

		if len(line) == 0 {
			continue
		}

		ctx := context.Background()
		response, err := ragApp.Handler.Process(ctx, line)
		if err != nil {
			slog.Error("process error", "error", err)
			continue
		}

		if response == nil {
			// Notification handled, no response needed
			continue
		}

		respBytes, _ := json.Marshal(response)
		fmt.Printf("%s\n", string(respBytes))
	}
}
