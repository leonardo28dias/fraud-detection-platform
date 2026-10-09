"""Tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_transaction():
    return {
        "Time": 100000.0,
        "Amount": 50.0,
        "V1": -0.5,
        "V2": 0.2,
        "V3": 1.1,
        "V4": -0.3,
        "V5": 0.8,
        "V6": -0.2,
        "V7": 0.5,
        "V8": 0.1,
        "V9": -0.4,
        "V10": 0.3,
        "V11": -0.6,
        "V12": 0.2,
        "V13": -0.5,
        "V14": 0.7,
        "V15": -0.2,
        "V16": 0.4,
        "V17": -0.3,
        "V18": 0.6,
        "V19": -0.1,
        "V20": 0.2,
        "V21": -0.4,
        "V22": 0.5,
        "V23": -0.2,
        "V24": 0.3,
        "V25": -0.5,
        "V26": 0.1,
        "V27": -0.3,
        "V28": 0.2,
    }


def test_health_check(client):
    """Health endpoint should always return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "threshold" in body


def test_predict_without_model_returns_503(client, monkeypatch, valid_transaction):
    """If the model artifact is not loaded, /predict should return 503."""
    import src.api.main as main_mod

    monkeypatch.setattr(main_mod, "pipeline", None)
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


def test_predict_missing_field_returns_422(client):
    """Missing a required V-feature should return 422 (validation error)."""
    payload = {"Time": 100000.0, "Amount": 50.0, "V1": -0.5, "V2": 0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_amount_returns_422(client):
    """Amount must be >= 0 per the schema."""
    payload = {
        "Time": 100000.0,
        "Amount": -50.0,
        "V1": -0.5,
        "V2": 0.2,
        "V3": 1.1,
        "V4": -0.3,
        "V5": 0.8,
        "V6": -0.2,
        "V7": 0.5,
        "V8": 0.1,
        "V9": -0.4,
        "V10": 0.3,
        "V11": -0.6,
        "V12": 0.2,
        "V13": -0.5,
        "V14": 0.7,
        "V15": -0.2,
        "V16": 0.4,
        "V17": -0.3,
        "V18": 0.6,
        "V19": -0.1,
        "V20": 0.2,
        "V21": -0.4,
        "V22": 0.5,
        "V23": -0.2,
        "V24": 0.3,
        "V25": -0.5,
        "V26": 0.1,
        "V27": -0.3,
        "V28": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_does_not_leak_internal_errors(client, monkeypatch, valid_transaction):
    """503 responses must not contain internal stack traces or file paths."""
    import src.api.main as main_mod

    monkeypatch.setattr(main_mod, "pipeline", None)
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "/app/" not in detail
    assert "Traceback" not in detail
    assert "Error" not in detail  # generic message only


def test_predict_success_when_model_loaded(client, valid_transaction):
    """When a model is loaded, /predict should return a valid prediction."""
    import src.api.main as main_mod

    if main_mod.pipeline is None:
        pytest.skip("Model artifact not available — run the pipeline first")
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert "transaction_id" in body
    assert "applied_threshold" in body
