# Makefile — convenience targets for the ML inference project
# Run `make help` to see all targets.

.PHONY: help install train test lint build run stop logs clean

# Use pip/python from the active venv if present, otherwise fall back to system
PYTHON   := $(shell command -v python3 2>/dev/null)
PIP      := $(shell command -v pip3 2>/dev/null || command -v pip 2>/dev/null)
IMAGE    := regression-inference
VERSION  ?= 1

help:
	@echo ""
	@echo "  make install   Install all dependencies"
	@echo "  make train     Train the model and log to MLflow"
	@echo "  make test      Run the full test suite"
	@echo "  make lint      Run flake8 + black"
	@echo "  make build     Build the Docker image"
	@echo "  make run       Start the inference service locally"
	@echo "  make stop      Stop and remove the running container"
	@echo "  make logs      Tail container logs"
	@echo "  make clean     Remove generated files"
	@echo ""

install:
	$(PIP) install -r train/requirements.txt
	$(PIP) install -r inference/requirements.txt
	$(PIP) install pytest pytest-cov flake8 black

train:
	$(PYTHON) train/train.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=inference --cov=train --cov-report=term-missing

lint:
	$(PYTHON) -m flake8 train/ inference/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	$(PYTHON) -m black --check --line-length=100 train/ inference/ tests/

format:
	$(PYTHON) -m black --line-length=100 train/ inference/ tests/

build:
	docker build -t $(IMAGE):$(VERSION) -f inference/Dockerfile .

run: build
	docker-compose up -d
	@echo ""
	@echo "  Service running → http://localhost:8080"
	@echo "  Health check   → http://localhost:8080/health"
	@echo "  Predict        → POST http://localhost:8080/predict"
	@echo ""

stop:
	docker-compose down

logs:
	docker-compose logs -f inference

clean:
	rm -rf mlruns/
	rm -f models/regression_model.joblib models/model_meta.json
	rm -f data/dummy_data.csv
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
	find . -name "coverage.xml" -delete
