"""Test fixtures and sample data for testing."""

import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
    })


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db'
        },
        'api': {
            'base_url': 'https://api.example.com',
            'timeout': 30
        }
    }


class MockData:
    """Mock data generators for testing."""
    
    @staticmethod
    def generate_user_data(num_records=100):
        """Generate mock user data."""
        import random
        names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
        cities = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin']
        
        return pd.DataFrame({
            'user_id': range(1, num_records + 1),
            'name': [random.choice(names) for _ in range(num_records)],
            'age': [random.randint(18, 65) for _ in range(num_records)],
            'city': [random.choice(cities) for _ in range(num_records)]
        })
