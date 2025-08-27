# Makefile for Data Engineering Project

.PHONY: help install install-dev test test-unit test-integration lint format clean setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests only
	pytest tests/integration/

lint: ## Run linting checks
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

setup: install-dev ## Complete project setup
	@echo "Project setup complete!"
	@echo "Run 'make help' to see available commands"

build: ## Build the package
	python -m build

notebook: ## Start Jupyter Lab
	jupyter lab src/notebooks/

airflow-init: ## Initialize Airflow
	airflow db init
	airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com

airflow-webserver: ## Start Airflow webserver
	airflow webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	airflow scheduler
