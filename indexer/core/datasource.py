import os
import logging
from typing import Iterator, List, ContextManager, Optional, Any, Dict
from dataclasses import dataclass, field
from core.chunker import TokenChunker
from core.cfg import IndexingConfig

logger = logging.getLogger("indexer.datasource")

@dataclass
class DataEntry:
    embedding_text: str
    text: Optional[str] = None
    fts_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    index_key: Optional[str] = None  # Routing key for multi-index mode

class DataSource(ContextManager):
    """
    Generic contract for any data source.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self) -> Iterator[DataEntry]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns total record count (estimated as number of files)."""
        return 0

class FolderDataSource(DataSource):
    """
    A DataSource that scans a local directory for text files and chunks them using tokens.
    Initialized with IndexingConfig for chunking parameters.
    """
    def __init__(
        self, 
        config: IndexingConfig,
        tokenizer: Any
    ):
        self.target = config.target
        self.config = config
        self.tokenizer = tokenizer
        self.chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap
        )
        self.extensions = config.extensions
        self._files = self._collect_files()

    def _collect_files(self) -> List[str]:
        collected = []
        if not os.path.exists(self.target):
             logger.error(f"Directory not found: {self.target}")
             return []
             
        for root, _, files in os.walk(self.target):
            for file in files:
                if any(file.endswith(ext) for ext in self.extensions):
                    collected.append(os.path.join(root, file))
        
        logger.info(f"Found {len(collected)} files in {self.target}")
        return collected

    def __iter__(self) -> Iterator[DataEntry]:
        for file_path in self._files:
            try:
                logger.debug(f"Indexing file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        continue
                        
                    # Split into smart overlapping chunks based on tokens
                    chunks = self.chunker.split(content)
                    
                    filename = os.path.basename(file_path)
                    for i, chunk in enumerate(chunks):
                        # 1. Rich context for better vector search
                        embedding_text = f"File: {filename}\n{chunk}"
                        
                        # 2. Clean text for LLM to save tokens
                        text = chunk
                        
                        # 3. Structural metadata for system/UI
                        metadata = {
                            "filename": filename,
                            "path": file_path,
                            "chunk_index": i + 1,
                            "total_chunks": len(chunks)
                        }
                        
                        yield DataEntry(
                            embedding_text=embedding_text,
                            text=text,
                            metadata=metadata
                        )
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}", exc_info=True)

    def __len__(self) -> int:
        return len(self._files)