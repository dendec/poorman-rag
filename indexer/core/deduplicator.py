import hashlib
import string
import unicodedata
from typing import Optional, Set, Dict

class Deduplicator:
    def __init__(self, preload: Set[str] = None):
        self.seen_hashes: Set[str] = preload if preload else set()

    def is_duplicate(self, cleaned_text: str) -> bool:
        text_hash = hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

class TextProcessor:
    def __init__(self, chars_to_remove: str = None, normalization_mapping: Dict[str, str] = None):
        """
        Initializes the text processor with custom normalization rules.
        :param chars_to_remove: String of characters to remove during normalization.
        :param normalization_mapping: Dict of character replacements (e.g. {'ё': 'е'}).
        """
        if chars_to_remove is None:
            chars_to_remove = string.punctuation + string.whitespace + "«»—…“”"
        
        self.trans_table = str.maketrans('', '', chars_to_remove)
        
        # Apply custom mapping if provided
        if normalization_mapping:
            for src, dst in normalization_mapping.items():
                self.trans_table[ord(src)] = dst

    def clean(self, text: str) -> Optional[str]:
        """Performs advanced text cleanup and normalization for deduplication."""
        if not text:
            return None
        
        # 1. Unicode normalization (NFKD decomposes combined characters)
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Lowercase and translation table (removal + specific maps)
        cleaned = text.lower().translate(self.trans_table)
        
        # 3. Final validation
        if not any(c.isalnum() for c in cleaned):
            return None
        if len(cleaned) < 4:
            return None
            
        return cleaned