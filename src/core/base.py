"""Base classes for all components"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseComponent(ABC):
    """Base class for all platform components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def __enter__(self):
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
