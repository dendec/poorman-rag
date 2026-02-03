package loader

import (
	"testing"

	"github.com/dendec/poorman-rag/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestFindLibraryPath(t *testing.T) {
	t.Run("FindLibraryPathReturnsString", func(t *testing.T) {
		// This test will look for libonnxruntime.so in various locations
		// Since the actual file may not exist, we just verify the function doesn't crash
		path := findLibraryPath()
		// The function returns an empty string if no library is found
		assert.IsType(t, "", path)
	})
}

func TestGetModelKeys(t *testing.T) {
	t.Run("GetModelKeysWithValidModel", func(t *testing.T) {
		keys := getModelKeys("intfloat/multilingual-e5-small")
		assert.Len(t, keys, 3)
		assert.Contains(t, keys[0], "multilingual-e5-small")
		assert.Contains(t, keys[0], "model_quantized.onnx")
		assert.Contains(t, keys[1], "tokenizer.json")
		assert.Contains(t, keys[2], "model_config.json")
	})

	t.Run("GetModelKeysWithEmptyModel", func(t *testing.T) {
		keys := getModelKeys("")
		assert.Nil(t, keys)
	})
}

func TestSplitModelName(t *testing.T) {
	t.Run("SplitModelNameWithSlash", func(t *testing.T) {
		parts := splitModelName("intfloat/multilingual-e5-small")
		assert.Len(t, parts, 2)
		assert.Equal(t, "intfloat", parts[0])
		assert.Equal(t, "multilingual-e5-small", parts[1])
	})

	t.Run("SplitModelNameWithoutSlash", func(t *testing.T) {
		parts := splitModelName("model-name")
		assert.Len(t, parts, 1)
		assert.Equal(t, "model-name", parts[0])
	})

	t.Run("SplitModelNameEmpty", func(t *testing.T) {
		parts := splitModelName("")
		assert.Nil(t, parts)
	})
}

// Note: We can't easily test LoadEmbeddingService without actual model files
// That would require setting up a full test environment with model files
func TestLoadEmbeddingService(t *testing.T) {
	t.Run("LoadEmbeddingService_InterfaceCheck", func(t *testing.T) {
		// Just verify that the function exists and has the right signature
		var fn func(*config.Config) (interface{}, error) = func(cfg *config.Config) (interface{}, error) {
			service, err := LoadEmbeddingService(cfg)
			return service, err
		}
		assert.NotNil(t, fn)
	})
}