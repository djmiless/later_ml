# ML Inference — Take-Home Assessment

A minimal but production-oriented ML workflow for a regression model.
Covers training & versioning, model serving, CI/CD, and observability design.

---

## Quick start

```bash
# 0. Create and activate a virtual environment (required on modern macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 1. Install all dependencies
make install

# 2. Train the model (generates data, trains, logs to MLflow)
make train

# 3. Run the test suite
make test

# 4. Build the Docker image and start the API locally
make run

# 5. Test the API
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"age": 42, "income_k": 88.0, "tenure_years": 6}'
# → {"prediction": 64080.1421, "model_version": "1"}

# 6. View MLflow experiment UI (run in a separate terminal, venv active)
mlflow ui  # → http://localhost:5000
```

---

## Project structure

```
.
├── train/
│   ├── train.py              Training script with MLflow logging
│   └── requirements.txt
├── inference/
│   ├── app.py                Flask API (/predict, /health, /metrics)
│   ├── Dockerfile            Multi-stage Docker build
│   └── requirements.txt
├── tests/
│   ├── test_api.py           API unit + integration tests
│   └── test_train.py         Training pipeline tests
├── monitoring/
│   └── design.md             Observability design document
├── models/                   Generated model artifacts (git-ignored)
├── data/                     Generated dataset (git-ignored)
├── .github/workflows/
│   └── ci.yml                GitHub Actions CI/CD pipeline
├── docker-compose.yml        Local service stack
├── Makefile                  Convenience targets
└── README.md
```

---

## 1. Model Training & Versioning

**Script:** `train/train.py`

Generates a synthetic dataset (200 rows, 3 features), trains a `LinearRegression`
model, and logs everything to MLflow.

### What gets logged

| Type | Details |
|---|---|
| Parameters | model type, features, train/test split, coefficients, intercept |
| Metrics | MAE, RMSE, R² on held-out test set |
| Artifacts | `regression_model.joblib`, `dummy_data.csv` |
| Registry | model registered as `regression-model` in the MLflow Model Registry |

### Why MLflow over SageMaker Experiments

MLflow was chosen over SageMaker Experiments for two reasons:

1. **Local reproducibility** — MLflow runs on localhost with no cloud credentials.
   Anyone cloning this repo can run `python train/train.py` and inspect the full
   experiment history at `http://localhost:5000`. SageMaker requires an AWS account.

2. **Cloud portability** — MLflow is cloud-agnostic. It works the same whether you
   self-host it on EKS, use Databricks Managed MLflow, or run it locally. SageMaker
   Experiments locks you into AWS. If Later operates across cloud providers, this
   matters.

In a production AWS-native environment I'd use both: MLflow as the primary model
registry and experiment tracker, SageMaker Experiments for runs triggered directly
inside SageMaker Training Jobs (where the integration is native and free).

### Why the model is saved locally *and* logged to MLflow

The Flask API loads the model from a local file path at container startup. Depending
on a running MLflow server at inference time would couple the serving layer to the
tracking layer — two separate concerns. Logging to MLflow gives you the experiment
history; saving to disk gives the container a self-contained artifact. In production
you'd pull from S3 on startup, but the separation of concerns is the same.

---

## 2. Model Serving API

**File:** `inference/app.py`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Run inference. Returns prediction + model version. |
| `GET` | `/health` | Liveness probe. Returns `200` when model is loaded, `503` otherwise. |
| `GET` | `/metrics` | Request counters + latency. Replace with Prometheus client in production. |
| `GET` | `/` | API metadata. |

### Example request

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"age": 42, "income_k": 88.0, "tenure_years": 6}'
```

```json
{
  "prediction": 64080.1421,
  "model_version": "1"
}
```

### Input validation

All three fields are required and must be numeric. Values outside the training
distribution are rejected with a `400` and an explanation:

| Field | Valid range |
|---|---|
| `age` | 18 – 70 |
| `income_k` | 30 – 120 |
| `tenure_years` | 0 – 10 |

### Design decisions

- **`/health` endpoint** — not required by the spec, added because Kubernetes liveness
  probes and load balancers need a dedicated health signal. If the model file is missing
  or corrupt, this returns `503` before any real traffic reaches the container.

- **`MODEL_VERSION` as an env var** — the same Docker image runs in staging and
  production. CI/CD injects the version at deploy time. The version is always visible
  in the response payload, so every prediction is traceable back to the model that
  produced it.

- **Gunicorn in Docker** — Flask's built-in server is single-threaded and not safe for
  production. Gunicorn with 2 workers and 4 threads handles concurrent requests and
  restarts workers that crash without taking the service down.

- **Multi-stage Dockerfile** — the builder stage installs dependencies; the runtime
  stage copies only the installed packages. pip's build cache and any temporary
  build artefacts are not present in the final image, and no build tools ship to
  production.

- **Non-root user** — the container runs as `appuser`, not `root`. Required by most
  enterprise Kubernetes security policies (PodSecurityPolicy / OPA Gatekeeper).

### Running with Docker

```bash
# Build and start
docker-compose up --build

# Or manually
docker build -t regression-inference:1 -f inference/Dockerfile .
docker run -p 8080:8080 \
  -e MODEL_VERSION=1 \
  -v $(pwd)/models:/app/models:ro \
  regression-inference:1
```

---

## 3. CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

### Pipeline stages

```
push to main / PR
       │
   ┌───▼───┐
   │ lint  │  flake8 + black --check
   └───┬───┘
       │
   ┌───▼───┐
   │ test  │  pytest with coverage
   └───┬───┘
       │
   ┌───▼───┐
   │ train │  retrain model, upload joblib artifact  ← main branch only
   └───┬───┘
       │
   ┌───▼───┐
   │ build │  docker build + push to GHCR
   └───┬───┘
       │
   ┌───▼──────────┐
   │ deploy-stg   │  deploy to staging environment
   └───┬──────────┘
       │
   ┌───▼──────────┐
   │ integ-test   │  smoke test against staging endpoint
   └───┬──────────┘
       │ (tag push or manual trigger only)
   ┌───▼──────────┐
   │ promote-prod │  manual approval → canary 10% → 100%
   └──────────────┘
```

### Model promotion decision

Models are not automatically promoted to production on every merge. A model is promoted when:

1. A git tag (`v*.*.*`) is pushed — this is the signal that the team has validated the model.
2. A manual `workflow_dispatch` is triggered with `deploy_to_prod: true`.

The production job requires a **GitHub Environment approval** (configured in the repo settings),
meaning a second team member must approve before any traffic is shifted.

### Notes on the mock deploy steps

The `echo` statements in the deploy jobs are explicit placeholders for real cloud commands.
In a production environment they would be replaced with:

- `kubectl set image` (EKS) or `aws ecs update-service` (ECS)
- The structure is identical — only the last line changes

This was a deliberate choice: cloud credentials shouldn't be baked into a take-home
assessment, but the pipeline structure should reflect exactly how it would work in production.

---

## 4. Monitoring & Observability

See [`monitoring/design.md`](monitoring/design.md) for the full design.

Summary:

| Layer | Tooling | Key signals |
|---|---|---|
| API | Prometheus + Grafana | p50/p99 latency, error rate, RPS |
| Model performance | Airflow DAG + BigQuery + MLflow | MAE/RMSE vs. training baseline (daily) |
| Data drift | Evidently PSI + Airflow | PSI per feature: <0.1 OK, 0.1–0.2 warn, ≥0.2 retrain |

---

## What I would add with more time

1. **Real Prometheus + Grafana stack** — a `docker-compose` that spins up Prometheus
   and Grafana alongside the API so the monitoring layer is actually runnable locally.

2. **Model decoupled from the Docker image** — pull the model from S3 at container
   startup via `MODEL_PATH`. This means model updates don't require an image rebuild
   or a new deployment — just restart the pod with the new `MODEL_PATH` value.

3. **Shadow mode before canary** — the new model receives production traffic and logs
   predictions but doesn't serve responses. Quality is validated offline against ground
   truth before any user sees the new model. This is the step between staging and canary.

4. **Evidently as a real Airflow DAG** — the drift detection pseudocode in
   `monitoring/design.md` wired up as a runnable pipeline that publishes PSI scores
   to Grafana on a schedule.

5. **Model card** — a structured document recording what the model does, who it's for,
   known limitations, and fairness considerations. Required for enterprise ML governance.
