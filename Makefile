.PHONY: help setup lint test train-gnn quantum-label distill clean install-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  setup        - Install package in development mode with dev dependencies"
	@echo "  lint         - Run ruff linter and mypy type checker"
	@echo "  test         - Run pytest test suite with coverage"
	@echo "  train-gnn    - Train GNN surrogate model"
	@echo "  quantum-label - Run DMET+VQE for hard cases"
	@echo "  distill      - Train delta learning head"
	@echo "  clean        - Clean build artifacts and cache files"
	@echo "  install-dev  - Install development dependencies only"

# Setup development environment
setup: install-dev
	pip install -e .
	@echo "Development environment setup complete!"

# Install development dependencies
install-dev:
	pip install -e ".[dev]"
	@echo "Development dependencies installed!"

# Lint code with ruff and mypy
lint:
	@echo "Running ruff linter..."
	ruff check src/ tests/ scripts/
	@echo "Running mypy type checker..."
	mypy src/
	@echo "Linting complete!"

# Run tests with coverage
test:
	@echo "Running pytest test suite..."
	pytest tests/ -v --cov=src/dft_hybrid --cov-report=term-missing --cov-report=html
	@echo "Test coverage report generated in htmlcov/"

# Train GNN surrogate model
train-gnn:
	@echo "Training GNN surrogate model..."
	python scripts/train_gnn.py --data-path data/processed --model-path models/gnn.pt --ensemble
	@echo "GNN training complete!"

# Run quantum labeling for hard cases
quantum-label:
	@echo "Running DMET+VQE for hard cases..."
	python scripts/label_quantum.py --hard-cases-path data/hard_cases --output-path data/quantum_labels
	@echo "Quantum labeling complete!"

# Train delta learning head
distill:
	@echo "Training delta learning head..."
	python scripts/distill_delta.py --gnn-model-path models/gnn.pt --quantum-labels-path data/quantum_labels --delta-model-path models/delta.pt
	@echo "Delta learning complete!"

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Quick development workflow
dev: clean setup lint test
	@echo "Development workflow complete!"

# Full pipeline run
pipeline:
	@echo "Running full DFT→GNN→QNN pipeline..."
	python -m dft_hybrid.pipeline.run --input-path data/raw --output-path results/
	@echo "Pipeline complete!"



