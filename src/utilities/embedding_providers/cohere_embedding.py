from typing import List

import numpy as np
from src.schemas.embedding_provider import EmbeddingProvider


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self.model = model
            self._dimension = 1024  # Default for embed-english-v3.0
        except ImportError:
            raise ImportError("Please install cohere: pip install cohere")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        if not texts:
            return np.array([])
        
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return np.array(response.embeddings)
        except Exception as e:
            print(f"Error embedding texts: {str(e)}")
            return np.zeros((len(texts), self._dimension))
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self._dimension)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self.model