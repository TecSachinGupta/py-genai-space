"""Protocol definitions for components"""
from typing import Protocol, List, Any


class CompletionProvider(Protocol):
    """Protocol for completion providers"""
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""
    
    def embed(self, text: str) -> List[float]:
        """Embed single text"""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        ...
