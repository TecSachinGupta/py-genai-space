"""Common utility functions for data processing."""

import logging
import pandas as pd
from typing import Optional, List, Any, Dict
import os
from pathlib import Path


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_data_frame(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Validate DataFrame structure and content."""
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def create_database_connection(connection_string: str = None):
    """Create database connection (mock implementation)."""
    # This is a placeholder - implement actual database connection logic
    from unittest.mock import Mock
    return Mock()


def read_config_file(file_path: str) -> Dict[str, Any]:
    """Read configuration from YAML or JSON file."""
    import yaml
    import json
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def batch_process(items: List[Any], batch_size: int = 1000):
    """Process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """Retry function execution on failure."""
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
