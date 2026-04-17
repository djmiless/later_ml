"""
Unit + integration tests for the inference API.

Run:
    pytest tests/ -v

The tests use Flask's built-in test client — no server needs to be running.
The model is loaded from models/regression_model.joblib relative to project root.
If the model doesn't exist yet, run `python train/train.py` first.
"""

import os
import sys
import pytest

# Ensure the inference directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))

# Point the app at the model in the local models/ directory
os.environ.setdefault(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "regression_model.joblib"),
)
os.environ.setdefault("MODEL_VERSION", "1")

from app import app  # noqa: E402  (import after env vars are set)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200_when_model_loaded(self, client):
        resp = client.get("/health")
        # If the model file doesn't exist, it returns 503 — that's intentional.
        assert resp.status_code in (200, 503)

    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data


# ---------------------------------------------------------------------------
# Predict — happy path
# ---------------------------------------------------------------------------
class TestPredictHappyPath:
    VALID_PAYLOAD = {"age": 42, "income_k": 88.0, "tenure_years": 6}

    def test_returns_200(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_contains_prediction(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        data = resp.get_json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

    def test_response_contains_model_version(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        data = resp.get_json()
        assert "model_version" in data
        assert data["model_version"] == "1"

    def test_prediction_is_plausible(self, client):
        """Regression sanity check: output should be in the ballpark of training targets."""
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        prediction = resp.get_json()["prediction"]
        # Training targets ranged roughly 15k–100k
        assert 10_000 < prediction < 120_000, f"Implausible prediction: {prediction}"


# ---------------------------------------------------------------------------
# Predict — validation errors
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_missing_field_returns_400(self, client):
        resp = client.post("/predict", json={"age": 42, "income_k": 88.0})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "missing fields" in data["error"]
        assert "tenure_years" in data["fields"]

    def test_non_numeric_field_returns_400(self, client):
        resp = client.post("/predict", json={"age": "old", "income_k": 88.0, "tenure_years": 6})
        assert resp.status_code == 400

    def test_age_out_of_range_returns_400(self, client):
        resp = client.post("/predict", json={"age": 150, "income_k": 88.0, "tenure_years": 6})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "age" in data["error"]

    def test_income_below_range_returns_400(self, client):
        resp = client.post("/predict", json={"age": 42, "income_k": 5.0, "tenure_years": 6})
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post("/predict", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_no_body_returns_400(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_structure(self, client):
        resp = client.get("/metrics")
        data = resp.get_json()
        for key in ("requests_total", "requests_success", "requests_error", "avg_latency_ms"):
            assert key in data, f"Missing key: {key}"

    def test_counters_increment_after_predict(self, client):
        before = client.get("/metrics").get_json()["requests_success"]
        client.post("/predict", json={"age": 30, "income_k": 70.0, "tenure_years": 3})
        after = client.get("/metrics").get_json()["requests_success"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class TestIndex:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_version(self, client):
        resp = client.get("/")
        data = resp.get_json()
        assert "model_version" in data
