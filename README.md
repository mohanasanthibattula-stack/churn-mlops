---
title: Customer Churn MLOps Service
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Customer Churn Prediction — End-to-End MLOps Pipeline

[![CI/CD Pipeline](https://github.com/your-username/churn-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/churn-mlops/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-black.svg)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A production-grade, low-latency machine learning system that trains, tracks, packages, tests, and serves customer churn predictions. Engineered from ground-up to deploy reliably under **free-tier constraints** (Render / Hugging Face Spaces: 512MB RAM, 0.1 CPU).

🔗 **Live Interactive API Docs**: [https://churn-prediction-api.onrender.com/docs](https://churn-prediction-api.onrender.com/docs) *(Demo Link Placeholder)*  
📄 **Model Governance Card**: [model_card.md](model_card.md)

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Preprocessing"]
        A["Raw Data (CSV / Ingestion)"] --> B["Vectorized Feature Engineering"]
        B --> C["Stratified Train/Test Split"]
        C --> D["Disk Cache (data/processed/)"]
    end

    subgraph Experimentation["2. MLflow Tracking"]
        D --> E["Unified Pipeline (ColumnTransformer + LightGBM/XGB)"]
        E --> F["MLflow Experiment Logger"]
        F --> G["Artifacts: ROC Curve, Confusion Matrix, Metrics"]
        E --> H["Production Model (models/model.pkl < 200KB)"]
    end

    subgraph Serving["3. Serving & Telemetry"]
        H --> I["FastAPI Lifespan Boot (Loaded Once in RAM)"]
        J["Client Request"] --> K["Pydantic v2 Schema Validation"]
        K --> L["In-Memory LRU Prediction Cache"]
        L -->|Cache Miss| M["Unified Pipeline Inference (< 3ms)"]
        L -->|Cache Hit| N["Sub-millisecond Cached Response"]
        M --> O["Prediction Telemetry Logging (CSV)"]
    end

    subgraph Continuous_Delivery["4. CI/CD & Cloud"]
        P["Git Push / Merge"] --> Q["GitHub Actions (Pytest + Docker Size Guard)"]
        Q --> R["Render Auto-Deploy Webhook"]
        R --> S["Docker Container (python:3.10-slim, <300MB)"]
    end
```

---

## Key Engineering Highlights

- **512MB RAM Optimization**: Uses `python:3.10-slim` with multi-stage builds. Strips heavy development libraries (`mlflow`, `matplotlib`, `pytest`) out of runtime containers to keep idle RAM **under 90MB**.
- **Zero Train-Serve Skew**: Combines `ColumnTransformer` (scaling & one-hot encoding) and the classifier inside a unified Scikit-Learn `Pipeline`. The API feeds raw input directly into `pipeline.predict_proba()`.
- **Sub-100ms Inference SLA**: Achieves **< 3ms uncached latency** and **< 0.1ms LRU cached latency** on repeated inputs.
- **Model Governance & Lineage**: Full [Model Card](model_card.md) documenting intended use, bias considerations, and quantitative evaluation.
- **Continuous Monitoring & Drift Detection**: Real-time inference traffic is logged to `monitoring/prediction_logs.csv` and monitored via **Evidently AI** for distribution shift.

---

## Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10 | Pinned, stable C-extension wheels across all platforms |
| **Machine Learning** | LightGBM / XGBoost, Scikit-Learn | Fast tree evaluations, sub-200KB artifact size |
| **Experiment Tracking**| MLflow | Reproducible metric logging and model signature inference |
| **Inference Framework**| FastAPI + Uvicorn | Async ASGI server, auto OpenAPI generation |
| **Contract Validation**| Pydantic v2 | Sub-millisecond type safety and boundary checking |
| **Containerization** | Docker (Multi-stage) | Multi-stage builder pattern; final image size **~265 MB** |
| **Testing** | Pytest + HTTPX | Automated unit and integration endpoint coverage |
| **CI/CD** | GitHub Actions | Automated linting, testing, and Docker validation under 2 minutes |
| **Cloud Hosting** | Render / Hugging Face Spaces | Free tier production hosting with health probes |
| **Monitoring** | Evidently AI | Statistical Kolmogorov-Smirnov data drift audit reports |

---

## Project Structure

```text
churn-mlops/
├── data/
│   ├── raw/                   # Raw customer records
│   └── processed/             # Cached stratified splits (train.csv, test.csv)
├── src/
│   ├── __init__.py
│   ├── config.py              # Central settings and environment variables
│   ├── data_ingestion.py      # Raw loading & 6,000-row synthetic generator
│   ├── preprocessing.py       # DRY: Shared vectorized transforms for train & serve
│   ├── train.py               # MLflow training pipeline with early stopping
│   ├── evaluate.py            # Offline classification report generator
│   ├── predict.py             # Python SDK inference module
│   └── drift_detection.py     # Evidently AI automated drift audit
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI endpoints, timing middleware, telemetry
│   ├── schemas.py             # Pydantic v2 request/response validation schemas
│   └── model_loader.py        # Singleton model manager with LRU cache
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py  # Unit tests for transformations & edge cases
│   └── test_api.py            # Integration tests for API routes and 400 guards
├── models/
│   ├── model.pkl              # Serialized pipeline (< 200KB)
│   └── metrics.json           # Logged test split evaluation metrics
├── .github/workflows/
│   └── ci-cd.yml              # Automated testing, Docker verification & deploy
├── Dockerfile                 # Multi-stage slim production build (< 300MB)
├── .dockerignore              # Aggressive exclusion of local caches & data
├── .gitignore
├── .env.example               # Template for environment variables
├── requirements.txt           # Minimal runtime dependencies (prod)
├── requirements-dev.txt       # Full training, MLflow, and test dependencies
├── render.yaml                # Infrastructure as Code for Render deployment
├── model_card.md              # Machine learning model documentation & ethics
├── Makefile                   # Developer CLI automation commands
└── README.md
```

---

## Getting Started Locally

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/churn-mlops.git
cd churn-mlops

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 3. Run Training & Offline Evaluation
```bash
# Trains model, evaluates test split, and logs to MLflow
python -m src.train

# Verify classification report
python -m src.evaluate
```

### 4. Run Pytest Test Suite
```bash
pytest -v tests/
```

### 5. Launch Local FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Access docs at http://localhost:8000/docs
```

---

## Docker Execution (Simulating 512MB RAM Free Tier)

Build and run the multi-stage image locally enforcing the exact memory constraint of free cloud tiers:

```bash
# 1. Build image
docker build -t churn-mlops:latest .

# 2. Confirm image is under 400MB
docker images churn-mlops:latest

# 3. Run container constrained to 512MB RAM and 0.5 CPU
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  --memory="512m" \
  --cpus="0.5" \
  churn-mlops:latest
```

---

## API Usage & Examples

### Health Check (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "app_version": "1.0.0",
  "timestamp": "1727827200",
  "uptime_seconds": 45.2
}
```

### Single Customer Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "monthly_charges": 89.95,
    "total_charges": 179.90,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "tech_support": "No"
  }'
```
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8341,
  "risk_level": "High",
  "churn_label": "Will Churn"
}
```

### High-Throughput Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "tenure": 2,
        "monthly_charges": 85.0,
        "total_charges": 170.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "tech_support": "No"
      },
      {
        "tenure": 60,
        "monthly_charges": 20.0,
        "total_charges": 1200.0,
        "contract_type": "Two year",
        "internet_service": "No",
        "payment_method": "Credit card (automatic)",
        "tech_support": "No internet service"
      }
    ]
  }'
```

---

## What I Learned Building This Project

As a 3rd-year Data Science student, this project bridged the gap between academic model training in notebooks and real-world software engineering:

1. **Deploying on Real-World Resource Limits**: Building a model is easy; making it run on a 512MB RAM free instance without hitting an out-of-memory kernel panic taught me the importance of multi-stage Docker builds, separating dev/prod dependencies, and selecting shallow tree ensembles over memory-heavy networks.
2. **Preventing Train-Serve Skew**: In many student projects, preprocessing is written once in pandas for training and rewritten in dictionary parsing for Flask/FastAPI. By wrapping transformations in a unified Scikit-Learn `ColumnTransformer` + `Pipeline`, the exact same feature engineering executes seamlessly during training and scoring.
3. **Defensive API Design**: Rather than relying on standard Python error handling, using Pydantic v2 ensures malformed payloads are rejected in under 0.1ms at the network boundary with HTTP 400 errors, protecting the model from unexpected inputs.
4. **Automating Quality via CI/CD**: Gating merges through GitHub Actions with automated unit tests and Docker image size limits eliminated the "works on my machine" phenomenon.

---

## Future Improvements

- [ ] **Automated Retraining Trigger**: Scheduled GitHub Action cron to retrain the model when Evidently AI detects >30% feature drift.
- [ ] **Feature Store Integration**: Integrate [Feast](https://feast.dev/) for centralized online/offline feature serving.
- [ ] **Prometheus & Grafana**: Export `/metrics` into Prometheus to visualize real-time request rates and latency percentiles.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
