package main

import (
	"context"
	"encoding/json"
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

func main() {
	lambda.Start(handleRequest)
}
