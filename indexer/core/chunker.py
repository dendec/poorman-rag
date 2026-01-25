import re
from typing import List, Any

class TokenChunker:
    def __init__(self, tokenizer: Any, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Initializes the recursive token-based chunker.
        :param tokenizer: A transformers-compatible tokenizer with .encode() and .decode() methods.
        :param chunk_size: Maximum size of a chunk in tokens.
        :param chunk_overlap: Number of overlapping tokens between consecutive chunks.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Standard structural separators
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks using recursive splitting for better coherence.
        """
        if not text.strip():
            return []

        # 1. Get initial splits recursively
        final_pieces = self._recursive_split(text, self.separators)
        
        # 2. Merge pieces into chunks of desired size with overlap
        return self._merge_pieces(final_pieces)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text until pieces are small enough or no separators left."""
        if self._get_token_count(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: just return as is if no more separators (should not happen with "")
            return [text]

        separator = separators[0]
        new_separators = separators[1:]
        
        # Split by the current separator
        if separator == "":
            # Character-wise split if empty string
            splits = list(text)
        else:
             splits = text.split(separator)

        # Handle the case where the split separator was removed (we want to preserve structural info)
        # We re-attach separators to each piece for better context
        pieces = []
        for i, s in enumerate(splits):
            piece = s
            if i < len(splits) - 1:
                piece += separator
            
            if piece:
                # Recurse if piece is still too long
                if self._get_token_count(piece) > self.chunk_size:
                    pieces.extend(self._recursive_split(piece, new_separators))
                else:
                    pieces.append(piece)
        
        return pieces

    def _merge_pieces(self, pieces: List[str]) -> List[str]:
        """Merges small pieces into chunks of roughly chunk_size with overlap."""
        chunks = []
        current_chunk_pieces = []
        current_token_count = 0

        for piece in pieces:
            piece_tokens = self._get_token_count(piece)
            
            if current_token_count + piece_tokens > self.chunk_size:
                # Flush the current chunk
                if current_chunk_pieces:
                    chunks.append("".join(current_chunk_pieces))
                    
                    # Start next chunk with overlap
                    # We keep pieces from the tail of current_chunk_pieces until we reach overlap limit
                    overlap_pieces = []
                    overlap_tokens = 0
                    for p in reversed(current_chunk_pieces):
                        p_t = self._get_token_count(p)
                        if overlap_tokens + p_t <= self.chunk_overlap:
                            overlap_pieces.insert(0, p)
                            overlap_tokens += p_t
                        else:
                            break
                    
                    current_chunk_pieces = overlap_pieces
                    current_token_count = overlap_tokens

            current_chunk_pieces.append(piece)
            current_token_count += piece_tokens

        if current_chunk_pieces:
            chunks.append("".join(current_chunk_pieces))

        return chunks