from abc import ABC, abstractmethod
from typing import Any, Dict

class DocumentProcessor(ABC):
    """Abstract base class for document processing"""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        pass