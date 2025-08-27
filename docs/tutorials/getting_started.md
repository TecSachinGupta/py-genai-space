# Getting Started Guide

Welcome to the Data Engineering Toolkit! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd data-engineering-boilerplate
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
make install-dev
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Your First ETL Job

### 1. Create a Simple Job

```python
# src/jobs/my_first_job.py
from src.utilities.common import setup_logging

def main():
    logger = setup_logging(__name__)
    logger.info("Starting my first ETL job")
    
    # Your ETL logic here
    
    logger.info("Job completed successfully")

if __name__ == "__main__":
    main()
```

### 2. Run the Job

```bash
python -m src.jobs.my_first_job
```

## Next Steps

- Explore the `src/notebooks/` directory for data analysis examples
- Check out `src/jobs/sample_etl_job.py` for a complete example
- Read the [ETL Pipeline Guide](etl_pipeline_guide.md) for advanced patterns

## Getting Help

- Check the [Structure Guide](../STRUCTURE_GUIDE.md) for extending the repository
- Review the test files in `tests/` for examples
- Open an issue if you encounter problems
