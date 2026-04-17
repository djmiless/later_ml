"""
Training script for the regression model.

Generates a synthetic dataset, trains a LinearRegression model,
logs parameters/metrics/artifact to MLflow, and saves the model
locally for use by the inference container.

Usage:
    pip install -r train/requirements.txt
    mlflow ui &          # start the tracking server
    python train/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dummy_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "regression_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Generate & save the synthetic dataset
# ---------------------------------------------------------------------------
def generate_dataset(path: str, n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, size=n)
    income_k = rng.normal(70, 15, size=n).clip(30, 120)
    tenure_years = rng.integers(0, 10, size=n)

    y = 10_000 + 120 * age + 500 * income_k + 800 * tenure_years + rng.normal(0, 3_000, size=n)

    df = pd.DataFrame({"age": age, "income_k": income_k, "tenure_years": tenure_years, "target": y})
    df.to_csv(path, index=False)
    print(f"[data] saved {len(df)} rows to {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame):
    FEATURES = ["age", "income_k", "tenure_years"]
    TARGET = "target"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    params = {
        "model_type": "LinearRegression",
        "features": FEATURES,
        "test_size": 0.2,
        "random_state": 42,
        "fit_intercept": model.fit_intercept,
        "coef_age": float(model.coef_[0]),
        "coef_income_k": float(model.coef_[1]),
        "coef_tenure_years": float(model.coef_[2]),
        "intercept": float(model.intercept_),
    }

    return model, params, metrics


# ---------------------------------------------------------------------------
# 3. Log to MLflow + save locally
# ---------------------------------------------------------------------------
def log_and_save(model, params: dict, metrics: dict):
    # MLflow tracking URI defaults to ./mlruns (local)
    # In production, point this at your MLflow server:
    #   export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("regression-model")

    with mlflow.start_run(run_name="linear-regression-v1") as run:
        run_id = run.info.run_id

        # Log hyperparameters / training config
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        # Log the model artifact to MLflow (for experiment tracking)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="regression-model",
        )

        # Also log the raw joblib file as an extra artifact
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="joblib")

        # Log dataset as artifact for reproducibility
        mlflow.log_artifact(DATA_PATH, artifact_path="data")

        print(f"[mlflow] run_id  : {run_id}")
        print(f"[mlflow] MAE     : {metrics['mae']:.2f}")
        print(f"[mlflow] RMSE    : {metrics['rmse']:.2f}")
        print(f"[mlflow] R²      : {metrics['r2']:.4f}")

    # Write a small metadata file consumed by the Flask app
    # This avoids a hard dependency on the MLflow server at inference time.
    meta = {
        "run_id": run_id,
        "model_version": os.getenv("MODEL_VERSION", "1"),
        "model_type": params["model_type"],
        "features": params["features"],
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, float)},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[local] model  saved → {MODEL_PATH}")
    print(f"[local] meta   saved → {META_PATH}")
    return run_id


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset(DATA_PATH)
    model, params, metrics = train(df)
    run_id = log_and_save(model, params, metrics)
    print(f"\n[done] MLflow run {run_id} complete.")
    print("       Run `mlflow ui` and open http://localhost:5000 to inspect.")
