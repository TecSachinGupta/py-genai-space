"""Sample unit tests for the data engineering toolkit."""

import pytest
from src.utilities.common import validate_data_frame, safe_divide


class TestUtilities:
    """Test suite for utility functions."""
    
    def test_safe_divide_normal_case(self):
        """Test normal division operation."""
        result = safe_divide(10, 2)
        assert result == 5.0
    
    def test_safe_divide_by_zero(self):
        """Test division by zero handling."""
        result = safe_divide(10, 0)
        assert result == 0.0
    
    def test_validate_data_frame_valid(self):
        """Test DataFrame validation with valid data."""
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert validate_data_frame(df, required_columns=["a", "b"]) is True
    
    def test_validate_data_frame_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert validate_data_frame(df, required_columns=["a", "b"]) is False
