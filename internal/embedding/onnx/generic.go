package onnx

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"strings"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

type PoolingType string

const (
	PoolingMean PoolingType = "mean"
	PoolingCLS  PoolingType = "cls"
	PoolingLast PoolingType = "last"
)

type GenericEmbedder struct {
	session    *ort.DynamicSession[int64, float32]
	tokenizer  *tokenizers.Tokenizer
	dimensions uint
	pooling    PoolingType
	normalize  bool
	maxLen     int
	// Input configuration
	hasTokenTypeIDs bool
	hasPositionIDs  bool
}

type Config struct {
	ModelPath     string      `json:"-"`
	TokenizerPath string      `json:"-"`
	Dimensions    uint        `json:"dimensions"`
	Pooling       PoolingType `json:"pooling"`
	Normalize     bool        `json:"normalize"`
	MaxLen        int         `json:"max_seq_length"`
}

func LoadConfig(path string) (Config, error) {
	var cfg Config
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	err = json.Unmarshal(data, &cfg)
	return cfg, err
}

func NewGenericEmbedder(libPath string, cfg Config) (*GenericEmbedder, error) {
	if !ort.IsInitialized() {
		ort.SetSharedLibraryPath(libPath)
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ORT: %w", err)
		}
	}

	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model not found: %s", cfg.ModelPath)
	}

	// Introspect results to find what the model actually expects
	inputNames, outputNames, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get model info: %w", err)
	}

	// Filter our candidate inputs against what the model wants
	// We always need input_ids and attention_mask.
	// We define what we CAPABLE of providing:
	candidates := map[string]bool{
		"input_ids":      true,
		"attention_mask": true,
		"token_type_ids": true,
		"position_ids":   true,
	}

	finalInputs := []string{}
	hasTokenType := false
	hasPosition := false

	for _, info := range inputNames {
		name := info.Name
		if candidates[name] {
			finalInputs = append(finalInputs, name)
			if name == "token_type_ids" {
				hasTokenType = true
			}
			if name == "position_ids" {
				hasPosition = true
			}
		}
	}

	// Ensure we have at least the basics
	if len(finalInputs) == 0 {
		// Fallback just in case introspection returned weird empty names but model works?
		// unlikely, but let's stick to basics if empty
		finalInputs = []string{"input_ids", "attention_mask"}
	}

	// Since we got outputNames from the model, we can use them or stick to a default.
	// Usually "last_hidden_state" or "embeddings".
	// Let's verify if "last_hidden_state" is in the outputs, otherwise allow "embeddings" or take the first one.
	targetOutput := "last_hidden_state"
	foundOutput := false
	for _, outInfo := range outputNames {
		out := outInfo.Name
		if out == targetOutput {
			foundOutput = true
			break
		}
	}
	if !foundOutput && len(outputNames) > 0 {
		targetOutput = outputNames[0].Name
		slog.Info("default output not found, using first available", "original", "last_hidden_state", "new", targetOutput)
	}
	finalOutputs := []string{targetOutput}

	session, err := ort.NewDynamicSession[int64, float32](cfg.ModelPath, finalInputs, finalOutputs)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	tk, err := tokenizers.FromFile(cfg.TokenizerPath)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("tokens error: %w", err)
	}

	if cfg.MaxLen <= 0 {
		cfg.MaxLen = 512
	}

	return &GenericEmbedder{
		session:         session,
		tokenizer:       tk,
		dimensions:      cfg.Dimensions,
		pooling:         cfg.Pooling,
		normalize:       cfg.Normalize,
		maxLen:          cfg.MaxLen,
		hasTokenTypeIDs: hasTokenType,
		hasPositionIDs:  hasPosition,
	}, nil
}

func (e *GenericEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	uintIDs, _ := e.tokenizer.Encode(text, true)
	contentLen := len(uintIDs)
	if contentLen > e.maxLen {
		uintIDs = uintIDs[:e.maxLen]
		contentLen = e.maxLen
	}
	if contentLen == 0 {
		return make([]float32, e.dimensions), nil
	}

	seqLen := int64(contentLen)
	shape := []int64{1, seqLen}

	inputIDs := make([]int64, seqLen)
	attentionMask := make([]int64, seqLen)
	for i := 0; i < contentLen; i++ {
		inputIDs[i] = int64(uintIDs[i])
		attentionMask[i] = 1
	}

	tInput, _ := ort.NewTensor(shape, inputIDs)
	defer tInput.Destroy()
	tMask, _ := ort.NewTensor(shape, attentionMask)
	defer tMask.Destroy()

	inputs := []*ort.Tensor[int64]{tInput, tMask}

	if e.hasTokenTypeIDs {
		tType, _ := ort.NewTensor(shape, make([]int64, seqLen))
		defer tType.Destroy()
		inputs = append(inputs, tType)
	}
	if e.hasPositionIDs {
		posIDs := make([]int64, seqLen)
		for i := range posIDs {
			posIDs[i] = int64(i)
		}
		tPos, _ := ort.NewTensor(shape, posIDs)
		defer tPos.Destroy()
		inputs = append(inputs, tPos)
	}

	outputSize := 1 * seqLen * int64(e.dimensions)
	outputData := make([]float32, outputSize)
	tOutput, _ := ort.NewTensor([]int64{1, seqLen, int64(e.dimensions)}, outputData)
	defer tOutput.Destroy()

	err := e.session.Run(inputs, []*ort.Tensor[float32]{tOutput})
	if err != nil {
		return nil, err
	}

	var vec []float32
	switch strings.ToLower(string(e.pooling)) {
	case string(PoolingCLS):
		vec = make([]float32, e.dimensions)
		copy(vec, outputData[:e.dimensions])
	case string(PoolingLast):
		vec = make([]float32, e.dimensions)
		offset := (seqLen - 1) * int64(e.dimensions)
		copy(vec, outputData[offset:offset+int64(e.dimensions)])
	default: // Mean
		vec = e.meanPooling(outputData, attentionMask)
	}

	if e.normalize {
		e.doNormalize(vec)
	}

	return vec, nil
}

func (e *GenericEmbedder) meanPooling(rawData []float32, mask []int64) []float32 {
	hidden := int64(e.dimensions)
	vec := make([]float32, hidden)
	var sumMask float32
	for i, m := range mask {
		if m == 1 {
			sumMask += 1.0
			offset := int64(i) * hidden
			for j := 0; j < int(hidden); j++ {
				vec[j] += rawData[offset+int64(j)]
			}
		}
	}
	if sumMask > 0 {
		for j := range vec {
			vec[j] /= sumMask
		}
	}
	return vec
}

func (e *GenericEmbedder) doNormalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 1e-9 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}

func (e *GenericEmbedder) Close() error {
	e.tokenizer.Close()
	return e.session.Destroy()
}
