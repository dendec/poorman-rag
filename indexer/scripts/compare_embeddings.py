import os
import numpy as np
from openai import OpenAI
from core.embedder import Embedder as LocalEmbedder

# Settings
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL", "intfloat/multilingual-e5-small")
TEST_TEXT = "This is a test sentence for embedding comparison."

class RemoteEmbedder:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        # OpenAI API expects a list of strings
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float"
        )
        # Extract vectors and sort by index
        data = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in data]
        return np.array(vectors, dtype=np.float32)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    if not API_KEY:
        print("❌ Please set OPENAI_API_KEY env var")
        return

    print(f"🤖 Initializing Local Embedder ({MODEL_NAME})...")
    local = LocalEmbedder(MODEL_NAME, device="cpu")

    print(f"☁️ Initializing Remote Embedder ({BASE_URL})...")
    remote = RemoteEmbedder(API_KEY, BASE_URL, MODEL_NAME)

    print(f"\n🧪 Testing text: '{TEST_TEXT}'")

    # 1. Local
    vec_local = local.embed([TEST_TEXT])[0]
    
    # 2. Remote
    vec_remote = remote.embed([TEST_TEXT])[0]

    # 3. Compare
    similarity = cosine_similarity(vec_local, vec_remote)
    
    print(f"\n📊 Results:")
    print(f"Local vector shape: {vec_local.shape}")
    print(f"Remote vector shape: {vec_remote.shape}")
    print(f"Cosine Similarity: {similarity:.6f}")

    if similarity > 0.99:
        print("\n✅ SUCCESS: Vectors are identical (or very close).")
    else:
        print("\n⚠️ WARNING: Vectors differ significantly. Re-indexing might be required if switching inference providers.")

if __name__ == "__main__":
    main()