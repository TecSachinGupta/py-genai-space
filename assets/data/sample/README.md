# Sample Data

This directory contains sample datasets for testing and development.

## Files

- `sample_data.csv`: Sample user data with basic demographics
- Add more sample files as needed for your specific use cases

## Usage

```python
import pandas as pd

# Load sample data
df = pd.read_csv('assets/data/sample/sample_data.csv')
print(df.head())
```

## Guidelines

- Keep sample files small (<1MB) to avoid repository bloat
- Use realistic but anonymized data
- Document the structure and purpose of each dataset
