import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Any, TYPE_CHECKING
from transformers import AutoTokenizer, AutoModel

if TYPE_CHECKING:
    from core.cfg import IndexingConfig

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
    def __init__(
        self, 
        model_name: str, 
        device: str = "auto", 
        max_length: int = 512, 
        vector_dtype: str = "float32", 
        pooling_mode: str = "mean",
        compile_model: bool = False,
        load_model: bool = True
    ):
        self.model_name = model_name
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.max_length = max_length
        self.vector_dtype = vector_dtype.lower()
        self.pooling_mode = pooling_mode.lower()
        self.compile_model = compile_model
        
        logger.info(f"🚀 Initializing Embedder on {self.device}")

        # Resolve tokenizer: prefer local directory
        slug = self.model_name.split("/")[-1]
        tokenizer_source = self.model_name
        for candidate in [self.model_name, f"models/{slug}", f"indexer/models/{slug}"]:
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tokenizer_config.json")):
                tokenizer_source = candidate
                logger.info(f"Using local tokenizer: {candidate}")
                break
        
        self.token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=self.token, trust_remote_code=True)
        # Separate tokenizer instance for thread-safe use in producer thread.
        self._producer_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=self.token, trust_remote_code=True)
        
        self.model = None
        if load_model:
            logger.info(f"Embedder loading on {self.device} (compile={compile_model}, max_length={max_length})...")
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                token=self.token,
                trust_remote_code=True
            ).to(self.device).to(torch.float32)
            
            if compile_model and self.device == "cuda" and hasattr(torch, "compile"):
                try:
                    logger.info("🚀 Compiling model for optimized inference...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    logger.warning(f"Compile skipped: {e}")
            
            self.model.eval()

    @classmethod
    def from_config(cls, config: "IndexingConfig", **kwargs) -> "Embedder":
        """Factory method to create an Embedder from a config object."""
        return cls(
            model_name=config.model_name,
            device=kwargs.get("device", config.device if hasattr(config, "device") else "auto"),
            max_length=config.max_tokens,
            vector_dtype=config.vector_dtype,
            pooling_mode=config.pooling_mode,
            compile_model=config.compile_model,
            load_model=kwargs.get("load_model", config.enable_vector_index)
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"Model not loaded for embedder {self.model_name}. Cannot generate embeddings.")
            
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
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

        # Explicit cleanup to help CUDA GC
        del encoded
        del model_out
        if torch.isnan(emb).any():
            logger.error("NaN detected in embeddings!")
            emb = torch.nan_to_num(emb)

        emb = F.normalize(emb, p=2, dim=1, eps=1e-6)
        res = emb.cpu().float().numpy()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        if self.vector_dtype in ["int8", "i8"]:
            # Check for NaN again in numpy just in case
            if np.isnan(res).any():
                res = np.nan_to_num(res)
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