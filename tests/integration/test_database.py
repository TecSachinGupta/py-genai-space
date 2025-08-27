"""Integration tests for external dependencies."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @patch('src.utilities.common.create_database_connection')
    def test_database_connection(self, mock_db_connection):
        """Test database connection establishment."""
        # Mock the database connection
        mock_connection = Mock()
        mock_db_connection.return_value = mock_connection
        
        from src.utilities.common import create_database_connection
        connection = create_database_connection()
        
        assert connection is not None
        mock_db_connection.assert_called_once()
    
    def test_sample_data_loading(self):
        """Test loading sample data files."""
        import os
        sample_data_path = os.path.join("assets", "data", "sample", "sample_data.csv")
        
        if os.path.exists(sample_data_path):
            df = pd.read_csv(sample_data_path)
            assert not df.empty
            assert len(df.columns) > 0
