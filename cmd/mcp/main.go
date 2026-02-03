package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-lambda-go/lambda"
	"github.com/dendec/poorman-rag/internal/app"
)

var ragApp *app.App

func init() {
	app.InitSlog()
	var err error
	ragApp, err = app.NewApp("")
	if err != nil {
		slog.Error("failed to initialize rag app", "error", err)
		os.Exit(1)
	}
}

// Lambda handler for AWS Lambda
func handleRequest(ctx context.Context, request events.APIGatewayV2HTTPRequest) (events.APIGatewayV2HTTPResponse, error) {
	if request.RequestContext.HTTP.Method != "POST" {
		return events.APIGatewayV2HTTPResponse{StatusCode: 405, Body: "Method Not Allowed"}, nil
	}

	response, err := ragApp.Handler.Process(ctx, []byte(request.Body))
	if err != nil {
		slog.Error("request processing error", "error", err)
		return events.APIGatewayV2HTTPResponse{StatusCode: 500, Body: "Internal Error"}, nil
	}

	bodyBytes, _ := json.Marshal(response)

	return events.APIGatewayV2HTTPResponse{
		StatusCode: 200,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		Body: string(bodyBytes),
	}, nil
}

// Stdio handler for local stdio server
func stdioHandler() {
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

func main() {
	// Determine mode: Lambda or Stdio
	mode := os.Getenv("MODE")

	if mode == "lambda" {
		// AWS Lambda mode
		lambda.Start(handleRequest)
		return
	}

	// Stdio mode - parse flags first to check for config
	configPath := flag.String("config", "", "path to config.yaml file")
	flag.Parse()

	// Reinitialize app with config if provided
	if *configPath != "" {
		ragApp = nil
		app.InitSlog()
		var err error
		ragApp, err = app.NewApp(*configPath)
		if err != nil {
			slog.Error("failed to initialize rag app with config", "error", err)
			os.Exit(1)
		}
	}

	// Stdio mode
	stdioHandler()
}