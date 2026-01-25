package embedding

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type OpenAIEmbedder struct {
	client openai.Client
	model  string
}

// NewOpenAIEmbedder creates a new embedder compatible with OpenAI API.
func NewOpenAIEmbedder(apiKey, baseURL, model string) *OpenAIEmbedder {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)

	return &OpenAIEmbedder{
		client: client,
		model:  model,
	}
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := e.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.Opt(text),
		},
		Model:          e.model,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return nil, fmt.Errorf("api error: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("empty response")
	}

	vec64 := resp.Data[0].Embedding
	vec32 := make([]float32, len(vec64))
	for i, v := range vec64 {
		vec32[i] = float32(v)
	}
	return vec32, nil
}
