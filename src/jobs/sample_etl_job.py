"""Sample ETL job implementation."""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from src.utilities.common import setup_logging, validate_data_frame
from src.configs.settings import get_settings
from src.schemas.data_models import User, JobStatus


class SampleETLJob:
    """Sample ETL job for processing user data."""
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        self.settings = get_settings()
        self.job_id = f"sample_etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def run(self) -> JobStatus:
        """Execute the complete ETL pipeline."""
        start_time = datetime.now()
        status = JobStatus(
            job_id=self.job_id,
            status="running",
            start_time=start_time
        )
        
        try:
            self.logger.info(f"Starting ETL job: {self.job_id}")
            
            # Extract
            raw_data = self.extract_data()
            self.logger.info(f"Extracted {len(raw_data)} records")
            
            # Transform
            processed_data = self.transform_data(raw_data)
            self.logger.info(f"Transformed data: {len(processed_data)} records")
            
            # Validate
            if not self.validate_data(processed_data):
                raise ValueError("Data validation failed")
            
            # Load
            self.load_data(processed_data)
            
            # Update status
            status.status = "success"
            status.end_time = datetime.now()
            status.records_processed = len(processed_data)
            
            self.logger.info(f"ETL job completed successfully: {self.job_id}")
            
        except Exception as e:
            status.status = "failed"
            status.end_time = datetime.now()
            status.error_message = str(e)
            self.logger.error(f"ETL job failed: {self.job_id} - {e}")
            
        return status
    
    def extract_data(self) -> pd.DataFrame:
        """Extract data from source."""
        # Load sample data
        sample_data_path = Path("assets/data/sample/sample_data.csv")
        
        if sample_data_path.exists():
            df = pd.read_csv(sample_data_path)
        else:
            # Generate sample data if file doesn't exist
            self.logger.warning("Sample data file not found, generating mock data")
            df = self._generate_mock_data()
        
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the extracted data."""
        # Data transformations
        df = df.copy()
        
        # Clean names
        df['name'] = df['name'].str.strip().str.title()
        
        # Add derived columns
        df['age_group'] = df['age'].apply(self._categorize_age)
        df['name_length'] = df['name'].str.len()
        df['is_adult'] = df['age'] >= 18
        
        # Add processing timestamp
        df['processed_at'] = datetime.now()
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the processed data."""
        required_columns = ['user_id', 'name', 'age', 'city']
        
        if not validate_data_frame(df, required_columns):
            self.logger.error("Data validation failed: missing required columns")
            return False
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.error(f"Data validation failed: null values found - {null_counts}")
            return False
        
        # Check age range
        if (df['age'] < 0).any() or (df['age'] > 120).any():
            self.logger.error("Data validation failed: invalid age values")
            return False
        
        self.logger.info("Data validation passed")
        return True
    
    def load_data(self, df: pd.DataFrame) -> None:
        """Load data to target destination."""
        # For this example, we'll save to a processed data file
        output_path = Path(f"processed_data_{self.job_id}.csv")
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Data saved to: {output_path}")
        
        # In a real scenario, you would load to your target database
        # Example database load:
        # df.to_sql('users', connection, if_exists='replace', index=False)
    
    def _generate_mock_data(self, num_records: int = 100) -> pd.DataFrame:
        """Generate mock data for testing."""
        import random
        
        names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
        cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin']
        
        data = {
            'user_id': range(1, num_records + 1),
            'name': [random.choice(names) for _ in range(num_records)],
            'age': [random.randint(18, 65) for _ in range(num_records)],
            'city': [random.choice(cities) for _ in range(num_records)],
            'signup_date': ['2023-01-01' for _ in range(num_records)]
        }
        
        return pd.DataFrame(data)
    
    def _categorize_age(self, age: int) -> str:
        """Categorize age into groups."""
        if age < 18:
            return 'Minor'
        elif age < 30:
            return 'Young Adult'
        elif age < 50:
            return 'Adult'
        elif age < 65:
            return 'Middle Aged'
        else:
            return 'Senior'


def main():
    """Main execution function."""
    job = SampleETLJob()
    result = job.run()
    
    print(f"Job Status: {result.status}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    else:
        print(f"Records Processed: {result.records_processed}")


if __name__ == "__main__":
    main()
