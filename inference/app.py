"""
Flask inference API for the regression model.

Endpoints:
  POST /predict  — run inference, return prediction + model version
  GET  /health   — liveness probe for K8s / load balancers
  GET  /metrics  — lightweight request counters (Prometheus-style in prod)
  GET  /         — API info

Environment variables:
  MODEL_PATH     — path to the joblib model file (default: ../models/regression_model.joblib)
  MODEL_VERSION  — version string injected by CI/CD at deploy time (default: 1)
  PORT           — port to listen on (default: 8080)
"""

import os
import time
import logging
from collections import defaultdict

import joblib
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inference-api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "regression_model.joblib"),
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
FEATURES = ["age", "income_k", "tenure_years"]

# Training-distribution bounds — used for input validation.
# Values outside these ranges are technically extrapolation.
FEATURE_BOUNDS = {
    "age": (18, 70),
    "income_k": (30, 120),
    "tenure_years": (0, 10),
}

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s (version=%s)", MODEL_PATH, MODEL_VERSION)
except Exception as exc:
    model_load_error = str(exc)
    logger.error("Failed to load model: %s", exc)

# ---------------------------------------------------------------------------
# In-memory counters (replace with prometheus_client in production)
# ---------------------------------------------------------------------------
_counters: dict = defaultdict(int)
_latency_sum: float = 0.0


def _record(status: str, latency_ms: float):
    _counters["requests_total"] += 1
    _counters[f"requests_{status}"] += 1
    global _latency_sum
    _latency_sum += latency_ms


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "regression-inference-api",
            "model_version": MODEL_VERSION,
            "endpoints": ["/predict", "/health", "/metrics"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Liveness probe.  Returns 200 when the model is loaded, 503 otherwise.
    Kubernetes readiness / liveness probes call this endpoint.
    Load balancers use it to gate traffic to healthy pods only.
    """
    if model is None:
        return jsonify({"status": "unhealthy", "reason": model_load_error}), 503
    return jsonify({"status": "healthy", "model_version": MODEL_VERSION}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Lightweight request counters in plain JSON.

    In production this endpoint would be replaced with the Prometheus client:
        from prometheus_client import Counter, Histogram, generate_latest
        @app.route("/metrics")
        def metrics():
            return generate_latest(), 200, {"Content-Type": "text/plain"}

    Prometheus would scrape this every 15 s and Grafana would visualise it.
    """
    avg_latency = (
        _latency_sum / _counters["requests_total"] if _counters["requests_total"] > 0 else 0.0
    )
    return jsonify(
        {
            "requests_total": _counters["requests_total"],
            "requests_success": _counters["requests_200"],
            "requests_error": _counters["requests_400"] + _counters["requests_500"],
            "avg_latency_ms": round(avg_latency, 2),
            "model_version": MODEL_VERSION,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference.

    Request body (JSON):
        {
            "age":          42,
            "income_k":     88.0,
            "tenure_years": 6
        }

    Response (JSON):
        {
            "prediction":    64091.98,
            "model_version": "1"
        }
    """
    t0 = time.time()

    if model is None:
        _record("500", 0)
        return jsonify({"error": "model not loaded", "detail": model_load_error}), 500

    # --- Parse body ---
    body = request.get_json(silent=True)
    if not body:
        _record("400", 0)
        return jsonify({"error": "request body must be valid JSON"}), 400

    # --- Validate fields ---
    missing = [f for f in FEATURES if f not in body]
    if missing:
        _record("400", 0)
        return jsonify({"error": "missing fields", "fields": missing}), 400

    # --- Coerce and range-check values ---
    values = {}
    for feat in FEATURES:
        try:
            val = float(body[feat])
        except (TypeError, ValueError):
            _record("400", 0)
            return jsonify({"error": f"'{feat}' must be numeric"}), 400

        lo, hi = FEATURE_BOUNDS[feat]
        if not (lo <= val <= hi):
            _record("400", 0)
            return (
                jsonify(
                    {
                        "error": f"'{feat}' out of expected range [{lo}, {hi}]",
                        "received": val,
                    }
                ),
                400,
            )
        values[feat] = val

    # --- Inference ---
    try:
        X = pd.DataFrame([values])[FEATURES]
        prediction = float(model.predict(X)[0])
    except Exception as exc:
        logger.exception("Inference error")
        _record("500", (time.time() - t0) * 1000)
        return jsonify({"error": "inference failed", "detail": str(exc)}), 500

    latency_ms = (time.time() - t0) * 1000
    _record("200", latency_ms)

    logger.info(
        "predict | input=%s | prediction=%.2f | latency=%.2fms",
        values,
        prediction,
        latency_ms,
    )

    return jsonify(
        {
            "prediction": round(prediction, 4),
            "model_version": MODEL_VERSION,
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint (for local dev — Gunicorn is used inside Docker)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
