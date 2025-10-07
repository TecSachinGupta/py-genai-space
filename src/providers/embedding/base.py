from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_single_text(self, text: str) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass