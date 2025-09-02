import json
from typing import Optional

import numpy as np


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputing"""
    
    def __init__(self, cache_file: Optional[str] = None):
        self.cache = {}
        self.cache_file = cache_file
        self.load_cache()
    
    def _hash_text(self, text: str) -> str:
        """Create a hash for the text"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            return np.array(self.cache[text_hash])
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        text_hash = self._hash_text(text)
        self.cache[text_hash] = embedding.tolist()
        self.save_cache()
    
    def load_cache(self):
        """Load cache from file"""
        if self.cache_file:
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except FileNotFoundError:
                self.cache = {}