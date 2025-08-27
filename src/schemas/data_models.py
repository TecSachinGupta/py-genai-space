"""Data models and schema definitions using Pydantic."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class User(BaseModel):
    """User data model."""
    user_id: int = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., ge=0, le=120, description="Age of the user in years")
    city: str = Field(..., description="City where the user is located")
    signup_date: datetime = Field(..., description="Date when the user signed up")
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name must not be empty')
        return v.strip()


class DataPipelineConfig(BaseModel):
    """Configuration for data pipeline."""
    source_type: str = Field(..., description="Type of data source")
    source_config: dict = Field(default_factory=dict, description="Source configuration")
    target_type: str = Field(..., description="Type of data target")
    target_config: dict = Field(default_factory=dict, description="Target configuration")
    batch_size: int = Field(default=1000, gt=0, description="Batch processing size")
    retry_count: int = Field(default=3, ge=0, description="Number of retries on failure")


class JobStatus(BaseModel):
    """Job execution status model."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    start_time: datetime = Field(..., description="Job start time")
    end_time: Optional[datetime] = Field(None, description="Job end time")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    records_processed: int = Field(default=0, ge=0, description="Number of records processed")


class MLModelConfig(BaseModel):
    """Machine learning model configuration."""
    model_name: str = Field(..., description="Name of the ML model")
    model_type: str = Field(..., description="Type of ML model")
    parameters: dict = Field(default_factory=dict, description="Model hyperparameters")
    training_data_path: str = Field(..., description="Path to training data")
    model_output_path: str = Field(..., description="Path to save trained model")
    feature_columns: List[str] = Field(default_factory=list, description="Feature column names")
    target_column: str = Field(..., description="Target column name")
