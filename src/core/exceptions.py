"""Custom exception hierarchy"""


class PlatformError(Exception):
    """Base exception for platform"""
    pass


class ComponentNotInitializedError(PlatformError):
    """Raised when component used before initialization"""
    pass


class ProviderError(PlatformError):
    """Raised when provider encounters an error"""
    pass


class StorageError(PlatformError):
    """Raised when storage operation fails"""
    pass


class ValidationError(PlatformError):
    """Raised when validation fails"""
    pass
