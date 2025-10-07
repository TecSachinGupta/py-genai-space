"""Main application settings using Pydantic"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Application
    app_name: str = "AI Data Platform"
    environment: str = "development"
    debug: bool = False
    version: str = "0.1.0"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Provider Configuration
    default_completion_provider: str = "openai"
    default_embedding_provider: str = "openai"
    
    # Provider API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Storage
    database_url: str = "sqlite:///./app.db"
    redis_url: str = "redis://localhost:6379/0"
    vector_store_type: str = "chroma"
    
    # Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"


settings = Settings()
