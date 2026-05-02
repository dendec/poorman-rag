import argparse
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import json
import logging
import sys
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()

# Resolve local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.s3 import S3Uploader
from core.cfg import IndexingConfig

logger = logging.getLogger("export")

def setup_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def save_model_config(model, model_id, target_dir: Path, cfg=None):
    config = model.config
    hidden_size = getattr(config, "hidden_size", 
                  getattr(config, "dim", 1024 if "0.6B" in model_id else 384))
    
    metadata = {
        "model_id": model_id,
        "dimensions": hidden_size,
        "max_seq_length": getattr(config, "max_position_embeddings", 4096),
        "pooling": "last" if "qwen" in model_id.lower() else "mean",
        "normalize": True,
    }
    
    if cfg and hasattr(cfg, 'pooling_mode'):
        metadata["pooling"] = cfg.pooling_mode

    with open(target_dir / "model_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📄 Saved Go config to {target_dir / 'model_config.json'}")

def last_token_pooling(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    last_token_indices = np.sum(mask, axis=1) - 1
    return embeddings[np.arange(embeddings.shape[0]), last_token_indices]

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-9, np.inf)

class ExportWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        # We need the last_hidden_state
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

def manual_export(model_id: str, target_dir: Path):
    logger.info(f"🛠️ Manual export for {model_id}...")
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    # Force loading in FP32 for CPU compatibility in Lambda
    model = AutoModel.from_pretrained(model_id, token=token, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    
    wrapper = ExportWrapper(model)
    dummy_input = tokenizer("test query", return_tensors="pt")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = target_dir / "model.onnx"
    
    logger.info(f"📤 Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper, 
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14, 
        do_constant_folding=True
    )
    tokenizer.save_pretrained(target_dir)
    return model, tokenizer

def quantize_model(onnx_path: Path, target_path: Path) -> None:
    logger.info(f"⚖️ Quantizing {onnx_path.name} to INT8...")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(target_path),
        per_channel=True,
        weight_type=QuantType.QInt8
    )
    logger.info(f"✅ Quantized model saved to {target_path}")

def test_model(model_path: Path, tokenizer_path: Path, text: str, max_length: int = 512, model_id: str = "") -> None:
    logger.info(f"🧪 Testing model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    inputs = tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors="np")
    
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    
    onnx_inputs = {name: inputs[name] for name in input_names if name in inputs}
    if "token_type_ids" in input_names and "token_type_ids" not in onnx_inputs:
        onnx_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    
    outputs = session.run(None, onnx_inputs)
    last_hidden_state = outputs[0]

    pooling_mode = "last" if "qwen" in model_id.lower() else "mean"
    if pooling_mode == "last":
        emb = last_token_pooling(last_hidden_state, inputs["attention_mask"])[0]
    else:
        # Mean pooling
        mask_exp = np.expand_dims(inputs["attention_mask"], -1)
        summ = np.sum(last_hidden_state * mask_exp, axis=1)
        summed_mask = np.clip(np.sum(mask_exp, axis=1), 1e-9, np.inf)
        emb = (summ / summed_mask)[0]
        
    emb = normalize(emb)
    logger.info(f"Test Query: '{text}' (Pooling: {pooling_mode})")
    logger.info(f"Embedding shape: {emb.shape}")
    logger.info(f"Embedding (first 5): {emb[:5]}")
    logger.info(f"Norm: {np.linalg.norm(emb)}")
    return emb

def validate_quantization(fp32_emb, quant_emb):
    similarity = np.dot(fp32_emb, quant_emb) / (np.linalg.norm(fp32_emb) * np.linalg.norm(quant_emb))
    logger.info(f"⚖️ Quantization Validation (Cosine Similarity): {similarity:.6f}")
    if similarity < 0.98:
        logger.warning("⚠️ High quantization divergence detected!")
    else:
        logger.info("✅ Quantization accuracy is excellent.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Export and quantize HF models to ONNX.")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--model_id", type=str, help="Override model ID")
    parser.add_argument("--output_dir", type=str, default="models/onnx", help="Output directory")
    parser.add_argument("--test_text", type=str, default="query: hello world", help="Test text")
    
    args = parser.parse_args()
    setup_logging("INFO")

    cfg = IndexingConfig.from_yaml(args.config) if args.config else None
    model_id = args.model_id or (cfg.model_name if cfg else "Qwen/Qwen3-Embedding-0.6B")
    model_slug = model_id.split("/")[-1]
    
    target_dir = Path(args.output_dir) / model_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = target_dir / "model.onnx"
    quant_path = target_dir / "model_quantized.onnx"

    try:
        if not onnx_path.exists():
            model, tokenizer = manual_export(model_id, target_dir)
            save_model_config(model, model_id, target_dir, cfg)
        
        if not quant_path.exists():
            quantize_model(onnx_path, quant_path)
            # Remove the large FP32 ONNX file to save space in Lambda
            # onnx_path.unlink() 

        logger.info("🧪 Testing FP32 model...")
        fp32_emb = test_model(onnx_path, target_dir, args.test_text, model_id=model_id)
        
        logger.info("🧪 Testing Quantized INT8 model...")
        quant_emb = test_model(quant_path, target_dir, args.test_text, model_id=model_id)
        
        validate_quantization(fp32_emb, quant_emb)
        logger.info("🎉 Done!")
    except Exception as e:
        logger.critical(f"💥 Export failed: {e}", exc_info=True)
        sys.exit(1)