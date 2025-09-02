from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.schemas.embedding_provider import EmbeddingProvider

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Hugging Face embedding provider using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"Error embedding texts: {str(e)}")
            return np.zeros((len(texts), self._dimension))
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Embed a single text"""
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            print(f"Error embedding text: {str(e)}")
            return np.zeros(self._dimension)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name