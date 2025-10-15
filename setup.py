"""
AI Data Platform Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-data-platform",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Universal AI and Data Engineering Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TecSachinGupta/py-genai-space",
    project_urls={
        "Bug Tracker": "https://github.com/TecSachinGupta/py-genai-space/issues",
        "Documentation": "https://tecsachingupta.github.io/py-genai-space/",
        "Source Code": "https://github.com/TecSachinGupta/py-genai-space",
    },
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.5.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-data-platform=applications.api.main:main",
        ],
    },
)