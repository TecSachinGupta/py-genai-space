# API Documentation

Auto-generated and manual API documentation.

## Structure

- `auto_generated/`: Documentation generated from code
- `manual/`: Hand-written API documentation

## Generation

Use Sphinx to generate documentation:

```bash
# Install Sphinx
pip install sphinx

# Generate docs
sphinx-quickstart docs/
sphinx-build -b html docs/ docs/_build/
```

## Guidelines

- Keep docstrings up to date
- Include examples in API documentation
- Regenerate docs before releases
