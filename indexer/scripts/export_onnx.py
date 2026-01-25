import argparse
import onnxruntime as ort
import numpy as np
import json
import logging
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Import local modules
# We adjust sys.path to find core.* if run from scripts/
sys.path.append(str(Path(__file__).parent.parent))
from core.s3 import S3Uploader
from core.cfg import IndexingConfig

logger = logging.getLogger("export")

def setup_logging(level_name: str):
    """Configures the logging system."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ---------- helpers ----------
def save_model_config(model, tokenizer, model_id, target_dir: Path):
    """Saves model metadata for Go server initialization."""
    config = model.config
    
    # Common hidden size/dimension keys
    hidden_size = getattr(config, "hidden_size", 
                  getattr(config, "dim", 
                  getattr(config, "model_max_length", 384)))
    
    # Try to find dimension in output shape if not in config
    if hasattr(model, "config") and hasattr(model.config, "pooler_type"):
         pooling = model.config.pooler_type
    else:
         pooling = "mean" # Default

    metadata = {
        "model_id": model_id,
        "dimensions": hidden_size,
        "max_seq_length": getattr(config, "max_position_embeddings", 512),
        "pooling": pooling,
        "normalize": True, # Usually True for embeddings
    }
    
    # Specific overrides for known models
    if "e5" in model_id.lower():
        metadata["pooling"] = "mean"
    elif "qwen" in model_id.lower():
        metadata["pooling"] = "mean"
        metadata["max_seq_length"] = 32768 if "0.6B" in model_id else 8192
        # Qwen-Embedding-0.6B has 1024 dims
        if "0.6B" in model_id: metadata["dimensions"] = 1024

    with open(target_dir / "model_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📄 Saved Go config to {target_dir / 'model_config.json'}")

def mean_pooling(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_exp = np.expand_dims(mask, -1)
    summ = np.sum(embeddings * mask_exp, axis=1)
    summed_mask = np.clip(np.sum(mask_exp, axis=1), 1e-9, np.inf)
    return summ / summed_mask

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-9, np.inf)

def load_or_export_model(model_id: str, target_dir: Path):
    """Loads a model, trying different options (ONNX repo, Auto-export, Manual-export)."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 1. Check for pre-exported ONNX weights
    for subfolder in ["onnx", ""]:
        try:
            logger.info(f"🔍 Checking for pre-exported ONNX in '{subfolder or 'root'}'...")
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_id, subfolder=subfolder, export=False, trust_remote_code=True
            )
            logger.info(f"✅ Found pre-exported ONNX in '{subfolder}'")
            return model, tokenizer
        except Exception:
            continue

    # 2. Attempt automatic export via Optimum
    try:
        logger.info(f"🚀 Attempting automatic export for {model_id}...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_id, export=True, trust_remote_code=True
        )
        logger.info("✅ Automatic export success")
        return model, tokenizer
    except Exception as e:
        logger.warning(f"⚠️ Automatic export failed: {e}")

    # 3. Manual export as a last resort
    logger.info("🛠️ Falling back to manual export...")
    try:
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hf_model.eval()
        
        class ExportWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask):
                out = self.m(input_ids=input_ids, attention_mask=attention_mask)
                return out.last_hidden_state

        wrapper = ExportWrapper(hf_model)
        dummy_input = tokenizer("test", return_tensors="pt")
        
        target_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = target_dir / "model.onnx"
        
        torch.onnx.export(
            wrapper, (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14, do_constant_folding=True
        )
        tokenizer.save_pretrained(target_dir)
        model = ORTModelForFeatureExtraction.from_pretrained(target_dir)
        logger.info("✅ Manual export + load success")
        return model, tokenizer
    except Exception as e:
        logger.error(f"❌ Manual export failed: {e}")
        return None, None

def quantize_model(model, tokenizer, target_dir: Path) -> None:
    """Quantize to INT8."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    logger.info(f"⚖️ Quantizing to INT8...")
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    
    quantizer = ORTQuantizer.from_pretrained(model)
    qcfg = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=target_dir, quantization_config=qcfg)
    logger.info("✅ Quantized to INT8")

    # Remove heavy FP32 files
    for f in target_dir.glob("model.onnx*"):
        f.unlink()
        logger.debug(f"♻️ Removed {f.name}")

def test_model(model_path: Path, tokenizer_path: Path, text: str, max_length: int = 512) -> None:
    logger.info(f"🧪 Testing model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    inputs = tokenizer(text,
                       max_length=max_length,
                       padding=True,
                       truncation=True,
                       return_tensors="np")
    
    session = ort.InferenceSession(str(model_path))
    input_names = [i.name for i in session.get_inputs()]
    
    onnx_inputs = {}
    for name in input_names:
        if name in inputs:
            onnx_inputs[name] = inputs[name]
        elif name == "token_type_ids":
            onnx_inputs[name] = np.zeros_like(inputs["input_ids"])
        elif name == "position_ids":
            seq_len = inputs["input_ids"].shape[1]
            onnx_inputs[name] = np.arange(seq_len).reshape(1, -1).astype(np.int64)
    
    outputs = session.run(None, onnx_inputs)
    last_hidden_state = outputs[0]

    emb = mean_pooling(last_hidden_state, inputs["attention_mask"])[0]
    emb = normalize(emb)

    logger.info(f"Test Query: '{text}'")
    logger.info(f"Embedding shape: {emb.shape}")
    logger.info(f"Embedding (first 5): {emb[:5]}")
    logger.info(f"Norm: {np.linalg.norm(emb)}")

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and quantize HF models to ONNX.")
    parser.add_argument("--config", type=str, help="Path to YAML config (populates defaults)")
    parser.add_argument("--model_id", type=str, help="Override HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="../../models", help="Base directory for exported models")
    parser.add_argument("--test_text", type=str, default="query: hello world", help="Override test text")
    parser.add_argument("--max_length", type=int, help="Override max sequence length")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--upload", action="store_true", help="Upload to S3 if bucket is configured")
    
    args = parser.parse_args()
    setup_logging(args.log_level)

    # --- Config Loading ---
    cfg = None
    if args.config:
        logger.info(f"📂 Loading defaults from {args.config}")
        cfg = IndexingConfig.from_yaml(args.config)
        
    # Merge defaults from config
    model_id = args.model_id or (cfg.model_name if cfg else "intfloat/multilingual-e5-small")
    max_len  = args.max_length or (cfg.max_tokens if cfg else 512)
    s3_bucket = cfg.s3_bucket if cfg else None
    
    # We strictly use settings from config for S3 to avoid CLI clutter
    s3_endpoint = cfg.s3_endpoint if cfg else None
    s3_region = cfg.s3_region if cfg else "us-east-1"
    s3_access = cfg.s3_access_key if cfg else None
    s3_secret = cfg.s3_secret_key if cfg else None

    model_slug = model_id.split("/")[-1]
    model_slug = model_slug.replace("-ONNX", "").replace("-onnx", "")
    
    target_dir = Path(args.output_dir) / model_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    
    quant_model_path = target_dir / "model_quantized.onnx"

    try:
        if not quant_model_path.exists():
            model, tokenizer = load_or_export_model(model_id, target_dir)
            if model:
                quantize_model(model, tokenizer, target_dir)
                save_model_config(model, tokenizer, model_id, target_dir)
            else:
                logger.error("💥 Failed to load or export model.")
                sys.exit(1)
        else:
            logger.info(f"✨ Quantized model already exists in {target_dir}")
            if not (target_dir / "model_config.json").exists():
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                logger.info("Generating model_config.json for existing model...")
                model = ORTModelForFeatureExtraction.from_pretrained(target_dir)
                tokenizer = AutoTokenizer.from_pretrained(target_dir)
                save_model_config(model, tokenizer, model_id, target_dir)

        test_model(quant_model_path, target_dir, args.test_text, max_length=max_len)
        
        # --- Automated S3 Upload ---
        if args.upload and s3_bucket:
             uploader = S3Uploader(
                 bucket=s3_bucket,
                 endpoint=s3_endpoint,
                 region=s3_region,
                 access_key=s3_access,
                 secret_key=s3_secret
             )
             logger.info(f"🚀 Starting automated upload for model: {model_slug}")
             for f in target_dir.iterdir():
                 if f.is_file():
                     uploader.upload_file(str(f), f"rag/models/{model_slug}/{f.name}")

        logger.info("🎉 Done")
    except Exception as e:
        logger.critical(f"💥 Export failed: {e}", exc_info=True)
        sys.exit(1)