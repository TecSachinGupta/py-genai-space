"""
AI Data Platform

Backward compatibility layer for existing imports
"""
import warnings


def _deprecation_warning(old_path: str, new_path: str):
    """Show deprecation warning"""
    warnings.warn(
        f"{old_path} is deprecated. Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Maintain old imports for backward compatibility
try:
    from src.storage.indexes.vector import *
    from src.providers.embedding import *
    from src.data.sources import *
except ImportError:
    pass  # New structure not fully in place yet

__version__ = "0.2.0"
