"""
Integration tests for FastAPI inference endpoints.
Covers happy paths, edge cases, Pydantic 400 validation failures,
batch processing, response headers, and telemetry endpoints.
"""

from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from src.train import get_lean_classifier

from app.main import app
from app.model_loader import model_manager
from src.data_ingestion import generate_synthetic_churn_data
from src.preprocessing import build_preprocessor, engineer_features


@pytest.fixture(scope="session", autouse=True)
def ensure_model_loaded_for_tests():
    """
    Ensures a valid ML pipeline is loaded into model_manager before running API tests.
    If the full production model is not yet trained, trains a fast minimal pipeline.
    """
    if not model_manager.is_loaded:
        # Create a fast minimal pipeline on synthetic data for test isolation
        df = generate_synthetic_churn_data(n_samples=100, random_state=42)
        df_featured = engineer_features(df)
        X = df_featured.drop(columns=["churn"])
        y = df_featured["churn"]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", get_lean_classifier()),
            ]
        )
        pipeline.fit(X, y)
        model_manager._pipeline = pipeline


@pytest.fixture
def client() -> TestClient:
    """Provides a FastAPI test client instance."""
    return TestClient(app)


@pytest.fixture
def valid_customer_payload() -> Dict[str, Any]:
    """Provides a valid customer payload."""
    return {
        "tenure": 6,
        "monthly_charges": 75.50,
        "total_charges": 453.00,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "tech_support": "No",
    }


def test_root_endpoint(client: TestClient):
    """Verifies root endpoint returns project metadata and status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert "health_check" in data


def test_health_endpoint(client: TestClient):
    """Verifies health check endpoint returns 200 and model loaded state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_metrics_endpoint(client: TestClient):
    """Verifies metrics telemetry endpoint returns valid counters."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_predictions" in data
    assert "avg_latency_ms" in data


def test_predict_single_success(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies happy path for single customer prediction."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert data["churn_label"] in ["Will Retain", "Will Churn"]


def test_response_process_time_header(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies custom latency header is injected by middleware."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers


def test_predict_single_invalid_contract_type(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies Pydantic returns 400 Bad Request on invalid category."""
    payload = valid_customer_payload.copy()
    payload["contract_type"] = "3-Year Contract"  # Invalid

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Validation Error"


def test_predict_single_negative_tenure(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies Pydantic returns 400 Bad Request on negative tenure."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = -10  # Invalid constraint ge=0

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Validation Error"


def test_predict_single_missing_field(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies Pydantic returns 400 Bad Request when required feature is missing."""
    payload = valid_customer_payload.copy()
    del payload["monthly_charges"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Validation Error"


def test_predict_batch_success(client: TestClient, valid_customer_payload: Dict[str, Any]):
    """Verifies batch inference endpoint with multiple customers."""
    second_customer = valid_customer_payload.copy()
    second_customer["tenure"] = 48
    second_customer["contract_type"] = "Two year"

    batch_payload = {"customers": [valid_customer_payload, second_customer]}

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["risk_level"] in ["Low", "Medium", "High"]


def test_predict_batch_empty_array(client: TestClient):
    """Verifies batch endpoint rejects empty arrays with 400 Bad Request."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 400
