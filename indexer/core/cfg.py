import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger("indexer.config")

@dataclass
class IndexingConfig:
    db_file: str = "index/dataset.sqlite"
    index_file: str = "index/vectors.usearch"
    dataset_repo: str = ""
    target: str = ""
    extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".markdown"])
    output_dir: str = ""  # If set, enables multi-index mode (one index per index_key)
    min_index_entries: int = 10  # Minimum records per index_key to create an index
    datasource: str = "core.datasource.FolderDataSource"
    # Indexing parameters
    checkpoint_period: int = 100
    min_tokens: int = 5
    max_tokens: int = 512
    batch_size: int = 32
    enable_vector_index: bool = True

    
    # Chunking parameters in tokens
    chunk_size: int = 400
    chunk_overlap: int = 50
    
    # Model settings
    vector_dim: int = 384
    model_name: str = "intfloat/multilingual-e5-small"
    prefix: str = "passage: "
    index_metric: str = "cos"
    vector_dtype: str = "f16"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    # FTS5 settings
    fts_mode: str = "speed"
    compile_model: bool = True
    query_prefix: str = "query: "
    pooling_mode: str = "mean" # "mean" or "last"

    # RRF (Hybrid Search) settings
    rrf_k: int = 60
    limit_vector: int = 20
    limit_fts: int = 20
    top_k: int = 10

    # Text normalization settings
    normalization_mapping: Optional[Dict[str, str]] = None
    chars_to_remove: Optional[str] = None

    # S3-compatible storage settings
    s3_bucket: str = ""
    kb_alias: str = ""
    s3_endpoint: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # File extensions to index
    extensions: list[str] = field(default_factory=lambda: [".txt", ".md", ".markdown"])
    
    def validate(self):
        """Validates configuration parameters."""
        if self.chunk_size > self.max_tokens:
             raise ValueError(f"chunk_size ({self.chunk_size}) cannot exceed model max_tokens ({self.max_tokens})")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    @staticmethod
    def from_yaml(path: str) -> 'IndexingConfig':
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get valid fields from dataclass annotations
        valid_keys = IndexingConfig.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        config = IndexingConfig(**filtered_data)
        config.validate()
        return config