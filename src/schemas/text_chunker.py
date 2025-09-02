from abc import ABC, abstractmethod
from typing import Any, Dict, List

class TextChunker(ABC):
    """Abstract base class for text chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass