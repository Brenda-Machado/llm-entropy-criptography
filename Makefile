# Project Settings
PYTHON ?= python3
VENV_DIR ?= .venv
PIP ?= $(VENV_DIR)/bin/pip
PYTHON_VENV ?= $(VENV_DIR)/bin/python

# Files
REQ_FILE ?= requirements.txt
DEV_REQ_FILE ?= requirements-dev.txt

# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make venv            - Create virtual environment"
	@echo "  make install         - Install project dependencies"
	@echo "  make dev-install     - Install development dependencies"
	@echo "  make update          - Update all dependencies"
	@echo "  make clean           - Remove virtual environment"
	@echo "  make run             - Run the app (app.py)"
	@echo "  make generate        - Generate training dataset"
	@echo "  make finetune        - Finetune the model (after generating dataset)"
	@echo "  make gen_report_data - Generate report data (100 samples)"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linter"

# Create virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

# Install dependencies
.PHONY: install
install: venv
	$(PIP) install -r $(REQ_FILE)
	@echo "Dependencies installed"

# Install dev dependencies
.PHONY: dev-install
dev-install: install
	$(PIP) install -r $(DEV_REQ_FILE)
	@echo "Development dependencies installed"

# Update all dependencies
.PHONY: update
update:
	$(PIP) install --upgrade -r $(REQ_FILE)
	@echo "Dependencies updated"

# Clean environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.backup" -delete
	@echo "Cleaned virtual environment and cache files"

# Run the app
.PHONY: run
run:
	$(PYTHON_VENV) src/app.py

# Generate dataset
.PHONY: generate
generate:
	$(PYTHON_VENV) src/generate_training_dataset.py

# Finetune model
.PHONY: finetune
finetune:
	$(PYTHON_VENV) src/finetune_model.py

# Generate report data
.PHONY: gen_report_data
gen_report_data:
	$(PYTHON_VENV) src/generate_report_data.py --samples 100 --nist-samples 10

# Generate report data (quick test)
.PHONY: gen_report_data_test
gen_report_data_test:
	$(PYTHON_VENV) src/generate_report_data.py --samples 10 --nist-samples 3

# Run NIST tests
.PHONY: nist_test
nist_test:
	$(PYTHON_VENV) src/test_nist_integration.py --mode single

# Run batch NIST tests
.PHONY: nist_batch
nist_batch:
	$(PYTHON_VENV) src/test_nist_integration.py --mode batch --num-keys 10

# Compare strategies
.PHONY: nist_compare
nist_compare:
	$(PYTHON_VENV) src/test_nist_integration.py --mode compare

# Run tests (if you have a test suite)
.PHONY: test
test:
	$(PYTHON_VENV) -m pytest tests/ -v

# Run linter
.PHONY: lint
lint:
	$(PYTHON_VENV) -m flake8 src/
	$(PYTHON_VENV) -m black --check src/

# Format code
.PHONY: format
format:
	$(PYTHON_VENV) -m black src/

# Check dependencies
.PHONY: check-deps
check-deps:
	$(PIP) check

# Freeze dependencies
.PHONY: freeze
freeze:
	$(PIP) freeze > requirements.txt
	@echo "Dependencies frozen to requirements.txt"

# Show installed packages
.PHONY: list
list:
	$(PIP) list

# Verify Ollama is running
.PHONY: check-ollama
check-ollama:
	@curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && \
		echo "✓ Ollama is running" || \
		echo "✗ Ollama is not running. Start it with: ollama serve"

# Full setup (venv + install + check)
.PHONY: setup
setup: venv install check-ollama
	@echo ""
	@echo "Setup complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "Then start the application with:"
	@echo "  make run"