"""Sample Airflow DAG for data processing pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd
import logging


# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'sample_etl_pipeline',
    default_args=default_args,
    description='A sample ETL pipeline DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['example', 'etl'],
)


def extract_data(**context):
    """Extract data from source."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data extraction...")
    
    # Simulate data extraction
    data = {
        'user_id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': [20 + (i % 50) for i in range(1, 101)],
        'city': ['New York', 'London', 'Tokyo'] * 34,
    }
    df = pd.DataFrame(data)
    
    # Save to temporary location
    df.to_csv('/tmp/extracted_data.csv', index=False)
    logger.info(f"Extracted {len(df)} records")
    return '/tmp/extracted_data.csv'


def transform_data(**context):
    """Transform the extracted data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data transformation...")
    
    # Load data from extract step
    df = pd.read_csv('/tmp/extracted_data.csv')
    
    # Apply transformations
    df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Middle' if x < 50 else 'Senior')
    df['name_length'] = df['name'].str.len()
    
    # Save transformed data
    df.to_csv('/tmp/transformed_data.csv', index=False)
    logger.info(f"Transformed {len(df)} records")
    return '/tmp/transformed_data.csv'


def load_data(**context):
    """Load data to target destination."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data loading...")
    
    # Load transformed data
    df = pd.read_csv('/tmp/transformed_data.csv')
    
    # Simulate loading to database or data warehouse
    # In a real scenario, you would load to your target system
    logger.info(f"Loaded {len(df)} records to target system")
    
    # Cleanup temporary files
    import os
    os.remove('/tmp/extracted_data.csv')
    os.remove('/tmp/transformed_data.csv')
    
    return f"Successfully processed {len(df)} records"


def validate_data(**context):
    """Validate the processed data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data validation...")
    
    # Implement data quality checks here
    # For example: check for null values, data types, ranges, etc.
    
    logger.info("Data validation completed successfully")
    return "Validation passed"


# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Task dependencies
extract_task >> transform_task >> load_task >> validate_task
