# Project Settings
PYTHON ?= python3.11
UV ?= uv
VENV_DIR ?= .venv

# Files
REQ_FILE ?= requirements.txt
DEV_REQ_FILE ?= requirements-dev.txt

# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make venv        - Create virtual environment using uv"
	@echo "  make install     - Install project dependencies"
	@echo "  make dev-install - Install development dependencies"
	@echo "  make update      - Update all dependencies"
	@echo "  make clean       - Remove virtual environment"
	@echo "  make run         - Run the app (app.py)"
	@echo "  make generate    - Generate training dataset"
	@echo "  make finetune    - Finetune the model (after generating dataset)"

# Create virtual environment
.PHONY: venv
venv:
	$(UV) venv --python $(PYTHON) $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

# Install dependencies
.PHONY: install
install:
	$(UV) pip install -r $(REQ_FILE)

# Install dev dependencies
.PHONY: dev-install
dev-install:
	$(UV) pip install -r $(REQ_FILE) -r $(DEV_REQ_FILE)

# Update all dependencies
.PHONY: update
update:
	$(UV) pip install --upgrade -r $(REQ_FILE)

# Clean environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	@echo "Removed virtual environment"

# Run the app
.PHONY: run
run:
	$(VENV_DIR)/bin/python src/app.py

# Generate dataset
.PHONY: generate
generate:
	$(VENV_DIR)/bin/python src/generate_training_dataset.py

# Finetune model
.PHONY: finetune
finetune:
	$(VENV_DIR)/bin/python src/finetune_model.py
