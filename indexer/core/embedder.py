import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("indexer.embedder")

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

class Embedder:
    def __init__(self, model_name: str, device: str | None = None, compile_model: bool = False, max_length: int = 512):
        """
        :param device: "cuda" or "cpu". If None - auto-select.
        :param compile_model: True for indexing (longer warm-up, faster execution).
                              False for search (instant start).
        :param max_length: Maximum context length.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Embedder loading on {self.device} (compile={compile_model}, max_length={max_length})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).to(torch.float16 if self.device == "cuda" else torch.float32)
        self.model.eval()
        self.max_length = max_length

        if compile_model and self.device == "cuda" and hasattr(torch, "compile"):
            try:
                logger.info("🚀 Compiling model for optimized inference...")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(f"Compile skipped (using default): {e}")

    def embed(self, texts: List[str]) -> np.ndarray:
        logger.debug(f"Computing embeddings for {len(texts)} texts")
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

        emb = _mean_pooling(model_out, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu().float().numpy()

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))