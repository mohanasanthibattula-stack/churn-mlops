.PHONY: help install install-dev train serve test docker-build docker-run clean

# Variables
PYTHON := python
PIP := pip
UVICORN := uvicorn
IMAGE_NAME := churn-mlops
IMAGE_TAG := latest
PORT := 8000

help:
	@echo "High-Efficiency Churn MLOps Commands:"
	@echo "  make install       : Install minimal production dependencies"
	@echo "  make install-dev   : Install development & training dependencies (MLflow, pytest)"
	@echo "  make train         : Run high-efficiency training pipeline (<500KB model artifact)"
	@echo "  make serve         : Launch local FastAPI server with LRU cache"
	@echo "  make test          : Execute pytest test suites"
	@echo "  make docker-build  : Build multi-stage slim Docker image (<350MB)"
	@echo "  make docker-run    : Run container with 512MB RAM restriction"
	@echo "  make clean         : Clean up bytecode and caches"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

train:
	$(PYTHON) -m src.train

serve:
	$(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	pytest -v tests/

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run -p $(PORT):$(PORT) --memory="512m" --cpus="0.5" --name $(IMAGE_NAME)-container --rm $(IMAGE_NAME):$(IMAGE_TAG)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
