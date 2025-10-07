"""
Tests for core base classes
"""

import pytest
from src.core.base import BaseComponent
from src.core.exceptions import ComponentNotInitializedError


class DummyComponent(BaseComponent):
    """Dummy component for testing"""
    
    def initialize(self):
        self._initialized = True
    
    def cleanup(self):
        self._initialized = False


class TestBaseComponent:
    """Test BaseComponent class"""
    
    def test_initialization(self):
        """Test component initialization"""
        component = DummyComponent()
        assert not component._initialized
        
        component.initialize()
        assert component._initialized
    
    def test_context_manager(self):
        """Test component as context manager"""
        with DummyComponent() as component:
            assert component._initialized
    
    def test_validate_not_initialized(self):
        """Test validation when not initialized"""
        component = DummyComponent()
        
        with pytest.raises(ComponentNotInitializedError):
            component.validate_initialized()
