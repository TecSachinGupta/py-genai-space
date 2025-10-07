"""Logging configuration for the application."""

import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def get_logging_config(log_level: str = "INFO", log_dir: str = "logs") -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    
    # Ensure log directory exists
    Path(log_dir).mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"app_{timestamp}.log")
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "src": {  # Application logger
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }


def setup_logging(log_level: str = None, log_dir: str = "logs") -> None:
    """Set up logging configuration."""
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    config = get_logging_config(log_level, log_dir)
    logging.config.dictConfig(config)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def get_logger(name: str, context: Dict[str, Any] = None) -> logging.Logger:
    """Get a logger with optional context."""
    logger = logging.getLogger(name)
    
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger
