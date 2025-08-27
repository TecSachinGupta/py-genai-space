"""Application settings and configuration management."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="", env="DB_USERNAME")
    password: str = Field(default="", env="DB_PASSWORD")
    database: str = Field(default="", env="DB_NAME")
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class APISettings(BaseSettings):
    """API configuration settings."""
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", env="API_BASE_URL")
    timeout: int = Field(default=30, env="API_TIMEOUT")
    max_retries: int = Field(default=3, env="API_MAX_RETRIES")


class SparkSettings(BaseSettings):
    """Spark configuration settings."""
    spark_home: str = Field(default="", env="SPARK_HOME")
    java_home: str = Field(default="", env="JAVA_HOME")
    master: str = Field(default="local[*]", env="SPARK_MASTER")
    app_name: str = Field(default="DataEngineering", env="SPARK_APP_NAME")
    executor_memory: str = Field(default="2g", env="SPARK_EXECUTOR_MEMORY")
    driver_memory: str = Field(default="1g", env="SPARK_DRIVER_MEMORY")


class CloudStorageSettings(BaseSettings):
    """Cloud storage configuration settings."""
    aws_access_key_id: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_DEFAULT_REGION")
    s3_bucket: str = Field(default="", env="S3_BUCKET_NAME")


class AppSettings(BaseSettings):
    """Main application settings."""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    spark: SparkSettings = SparkSettings()
    storage: CloudStorageSettings = CloudStorageSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment."""
    global _settings
    _settings = AppSettings()
    return _settings
