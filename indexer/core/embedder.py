import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("indexer.embedder")

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

def _last_token_pooling(model_output, attention_mask):
    # For Qwen and similar causal models, the embedding is usually the last non-masked token
    # or specifically trained to be at the last token position.
    token_embeddings = model_output[0]
    # Find the index of the last 1 in the mask for each sequence
    last_token_indices = attention_mask.sum(dim=1) - 1
    # Gather the embeddings at those indices
    return token_embeddings[torch.arange(token_embeddings.size(0)), last_token_indices]

class Embedder:
    def __init__(self, model_name: str, device: str | None = None, compile_model: bool = False, max_length: int = 512, vector_dtype: str = "float32", load_model: bool = True, pooling_mode: str = "mean"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.vector_dtype = vector_dtype.lower()
        self.pooling_mode = pooling_mode.lower()

        # Resolve tokenizer: prefer local directory (has tokenizer files), 
        # then fall back to HF cache (local_files_only=True, no network).
        slug = model_name.split("/")[-1]
        tokenizer_source = model_name  # default: HF cache lookup by name
        for candidate in [model_name, f"models/{slug}", f"indexer/models/{slug}"]:
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tokenizer_config.json")):
                tokenizer_source = candidate
                logger.info(f"Using local tokenizer: {candidate}")
                break

        logger.info(f"Embedder loading on {self.device} (compile={compile_model}, max_length={max_length})...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

        self.model = None
        if load_model:
            # Check if we should use the local path for weights too
            model_source = tokenizer_source
            weight_files = ["model.safetensors", "pytorch_model.bin", "model.pt"]
            is_local = os.path.isdir(tokenizer_source)
            has_weights = any(os.path.exists(os.path.join(tokenizer_source, f)) for f in weight_files)
            
            if is_local and not has_weights:
                logger.info(f"Local directory {tokenizer_source} has no PyTorch weights, falling back to HF Hub for model.")
                model_source = model_name

            # Load PyTorch model
            self.model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True
            ).to(self.device).to(torch.float16 if self.device == "cuda" else torch.float32)
            self.model.eval()

            if compile_model and self.device == "cuda" and hasattr(torch, "compile"):
                try:
                    logger.info("🚀 Compiling model for optimized inference...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    logger.warning(f"Compile skipped: {e}")

        # Separate tokenizer instance for thread-safe use in producer thread.
        # HuggingFace Fast Tokenizers (Rust-backed) are NOT thread-safe.
        self._producer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            # If vectors are disabled, return empty arrays (or zeros)
            return np.zeros((len(texts), 384), dtype=np.float32)
            
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    model_out = self.model(**encoded)
            else:
                model_out = self.model(**encoded)

        if self.pooling_mode == "last":
            emb = _last_token_pooling(model_out, encoded["attention_mask"])
        else:
            emb = _mean_pooling(model_out, encoded["attention_mask"])
            
        emb = F.normalize(emb, p=2, dim=1)
        res = emb.cpu().float().numpy()

        if self.vector_dtype in ["int8", "i8"]:
            return (res * 127).astype(np.int8)
        elif self.vector_dtype in ["float16", "f16"]:
            return res.astype(np.float16)

        return res

    def token_count(self, text: str) -> int:
        return len(self._producer_tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length * 2
        ))