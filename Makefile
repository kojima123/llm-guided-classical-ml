# LLM-Guided Classical ML - Reproducible Experiments

.PHONY: help install reproduce clean test lint format

# Default target
help:
	@echo "LLM-Guided Classical ML - Available Commands:"
	@echo ""
	@echo "  make install     - Install exact dependencies"
	@echo "  make reproduce   - Run all experiments with fixed seeds"
	@echo "  make test        - Run test suite"
	@echo "  make clean       - Clean generated files"
	@echo "  make lint        - Run code quality checks"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Environment:"
	@echo "  Python: $(shell python3 --version)"
	@echo "  OS: $(shell uname -s)"

# Install exact dependencies
install:
	@echo "Installing exact dependencies..."
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Reproduce all experiments
reproduce: install
	@echo "Starting reproducible experiments..."
	@echo "This will take approximately 10-15 minutes..."
	python -m experiments.run_all_experiments --seed 42 --trials 5
	python -m experiments.generate_figures
	@echo "✓ All experiments completed"
	@echo "Results saved in results/"

# Quick test run (faster for CI/development)
test-quick:
	@echo "Running quick test experiments..."
	python -m experiments.run_all_experiments --seed 42 --trials 1 --quick
	@echo "✓ Quick test completed"

# Run test suite
test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests completed. Coverage report in htmlcov/"

# Clean generated files
clean:
	rm -rf results/figures/*.png
	rm -rf results/data/*.csv
	rm -rf results/data/*.json
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf experiments/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✓ Cleaned generated files"

# Code quality
lint:
	flake8 src/ experiments/ --max-line-length=88
	black --check src/ experiments/
	isort --check-only src/ experiments/

format:
	black src/ experiments/
	isort src/ experiments/
	@echo "✓ Code formatted"

# Docker support
docker-build:
	docker build -t llm-guided-ml .

docker-run:
	docker run --rm -v $(PWD)/results:/app/results llm-guided-ml make reproduce

# Development setup
dev-install: install
	pip install black flake8 isort pytest pytest-cov
	@echo "✓ Development environment ready"
