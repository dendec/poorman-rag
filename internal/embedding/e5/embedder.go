package e5

import (
	"context"
	"fmt"
	"math"
	"os"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

const HiddenSize = 384

type Embedder struct {
	session   *ort.DynamicSession[int64, float32]
	tokenizer *tokenizers.Tokenizer
}

// NewEmbedder loads the model and tokenizer (tokenizer.json)
func NewEmbedder(libPath, modelPath, tokenizerPath string) (*Embedder, error) {
	// 1. ORT Initialization (if not already initialized)
	if !ort.IsInitialized() {
		ort.SetSharedLibraryPath(libPath)
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, fmt.Errorf("failed to init ORT: %w", err)
		}
	}

	// 2. File verification
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("onnx model not found: %s", modelPath)
	}
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer json not found: %s", tokenizerPath)
	}

	// 3. Session creation
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewDynamicSession[int64, float32](
		modelPath, inputNames, outputNames,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	// 4. Loading tokenizer from JSON
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &Embedder{
		session:   session,
		tokenizer: tk,
	}, nil
}

func (e *Embedder) Close() error {
	e.tokenizer.Close()
	return e.session.Destroy()
}

func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	// 1. Tokenization
	uintIDs, _ := e.tokenizer.Encode(text, true)

	contentLen := len(uintIDs)

	// Truncate to 512 tokens
	if contentLen > 512 {
		uintIDs = uintIDs[:512]
		contentLen = 512
	}

	seqLen := int64(contentLen)
	shape := []int64{1, seqLen} // Batch size 1

	inputIDs := make([]int64, seqLen)
	attentionMask := make([]int64, seqLen)
	tokenTypeIDs := make([]int64, seqLen) // Zeros

	for i := 0; i < contentLen; i++ {
		inputIDs[i] = int64(uintIDs[i])
		attentionMask[i] = 1 // 1 for real tokens
	}
	// 2. Tensor creation
	tInput, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, err
	}
	defer tInput.Destroy()

	tMask, err := ort.NewTensor(shape, attentionMask)
	if err != nil {
		return nil, err
	}
	defer tMask.Destroy()

	tType, err := ort.NewTensor(shape, tokenTypeIDs)
	if err != nil {
		return nil, err
	}
	defer tType.Destroy()

	// 3. Output tensor
	outputSize := 1 * seqLen * HiddenSize
	outputData := make([]float32, outputSize)
	tOutput, err := ort.NewTensor([]int64{1, seqLen, HiddenSize}, outputData)
	if err != nil {
		return nil, err
	}
	defer tOutput.Destroy()

	// 4. Run
	err = e.session.Run(
		[]*ort.Tensor[int64]{tInput, tMask, tType},
		[]*ort.Tensor[float32]{tOutput},
	)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// 5. Pooling & Norm
	vec := meanPooling(outputData, attentionMask, HiddenSize)
	normalize(vec)

	return vec, nil
}

// --- Helpers ---
func meanPooling(rawData []float32, mask []int64, hiddenSize int) []float32 {
	seqLen := len(mask)
	vector := make([]float32, hiddenSize)
	var sumMask float32 = 0.0

	for i := 0; i < seqLen; i++ {
		if mask[i] == 1 {
			sumMask += 1.0
			offset := i * hiddenSize
			for j := 0; j < hiddenSize; j++ {
				vector[j] += rawData[offset+j]
			}
		}
	}
	if sumMask > 0 {
		for j := 0; j < hiddenSize; j++ {
			vector[j] /= sumMask
		}
	}
	return vector
}

func normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}
