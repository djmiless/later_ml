# Monitoring & Observability Design

## Overview

This document describes how I would monitor the regression inference service in production.
The design is broken into three layers: **API observability**, **model performance tracking**,
and **data/prediction drift detection**. Each section includes the tooling choice, the
specific signals I'd track, and how alerts are triggered.

The goal is to catch three classes of failure:
1. **Infrastructure failure** — the service is slow or down (API layer)
2. **Model degradation** — predictions are less accurate than at training time (performance layer)
3. **Distribution shift** — the real-world inputs no longer look like training data (drift layer)

---

## 1. API Latency & Error Rates

### Tooling
- **Prometheus** — metrics scraping (15 s interval)
- **Grafana** — dashboards and alerting
- **Prometheus client** — embedded in the Flask app (`/metrics` endpoint)

### Instrumentation

The Flask app already exposes a `/metrics` endpoint. In production, replace the in-memory
counters with the official Prometheus client:

```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total prediction requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

@app.before_request
def start_timer():
    g.start = time.time()

@app.after_request
def record_metrics(response):
    latency = time.time() - g.start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.path,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.observe(latency)
    return response

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

### Signals to track

| Signal | Warning | Critical |
|---|---|---|
| `p50` latency (`/predict`) | > 200 ms | > 500 ms |
| `p99` latency (`/predict`) | > 800 ms | > 2 s |
| Error rate (4xx + 5xx / total) | > 1% | > 5% |
| `/health` non-200 | any | any |
| Requests per second | —  | drop > 50% vs. rolling 1h avg |

### Grafana dashboard panels

```
Row 1: Request rate (RPS), Error rate (%)
Row 2: Latency p50 / p99 / p999 time series
Row 3: Status code breakdown (stacked bar)
Row 4: Container CPU / memory (from cAdvisor)
```

### Alerting

```yaml
# prometheus/alerts.yml
groups:
  - name: inference-api
    rules:
      - alert: HighErrorRate
        expr: |
          rate(inference_requests_total{status_code=~"5.."}[5m])
          / rate(inference_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5% for 2 minutes"

      - alert: HighP99Latency
        expr: |
          histogram_quantile(0.99,
            rate(inference_request_duration_seconds_bucket[5m])
          ) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 latency > 2 s for 5 minutes"

      - alert: ServiceDown
        expr: up{job="inference-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Inference service is unreachable"
```

---

## 2. Model Performance After Deployment

### The challenge

Unlike classification (where you can measure accuracy immediately), regression
performance requires **ground truth labels that arrive with a delay** — for example,
a revenue prediction is only verifiable after the billing cycle closes.

### Approach: Delayed label join

```
┌──────────────┐     log to      ┌─────────────┐
│  Inference   │ ──────────────► │  BigQuery   │  predictions table
│  service     │                 │  (or S3)    │
└──────────────┘                 └──────┬──────┘
                                        │  nightly join
                                  ┌─────▼──────┐
                                  │  Ground    │  actuals table (from
                                  │  truth     │  billing/CRM system)
                                  └──────┬─────┘
                                         │
                                   ┌─────▼──────┐
                                   │  Airflow   │  compute MAE/RMSE
                                   │  DAG       │  vs. baseline
                                   └─────┬──────┘
                                         │
                                   ┌─────▼──────┐
                                   │  MLflow    │  log production metrics
                                   │  + Grafana │  vs. training metrics
                                   └────────────┘
```

### Inference logging

Each prediction is logged asynchronously to avoid adding latency:

```python
import threading
import bigquery  # or boto3 for S3

def _log_prediction(payload: dict, prediction: float, model_version: str):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "age": payload["age"],
        "income_k": payload["income_k"],
        "tenure_years": payload["tenure_years"],
        "prediction": prediction,
    }
    # Fire-and-forget — don't block the response
    threading.Thread(target=_write_to_bq, args=(record,), daemon=True).start()
```

### Performance tracking DAG (Airflow pseudocode)

```python
@dag(schedule_interval="@daily")
def model_performance_check():

    @task
    def compute_metrics():
        sql = """
            SELECT
                AVG(ABS(p.prediction - a.actual)) AS mae,
                SQRT(AVG(POW(p.prediction - a.actual, 2))) AS rmse
            FROM predictions p
            JOIN actuals a USING (request_id)
            WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
        """
        results = bigquery_client.query(sql).result()
        return dict(results)

    @task
    def compare_to_baseline(metrics: dict):
        baseline_mae = 2800  # logged during training
        drift_ratio = metrics["mae"] / baseline_mae
        if drift_ratio > 1.3:
            alert("Model MAE has degraded 30% vs. training baseline — investigate or retrain")
        mlflow.log_metrics(metrics, run_name="production-monitoring")
```

---

## 3. Data Drift & Prediction Drift Detection

### Why this matters

The model was trained on a specific distribution of `age`, `income_k`, and `tenure_years`.
If the population of users sending requests changes (e.g., a new customer segment is onboarded),
the model's predictions may become unreliable even if no error is raised.

### Approach: Population Stability Index (PSI) with Evidently

**PSI** compares a feature's distribution at training time (baseline) vs. recently (window):
- PSI < 0.1 → no significant drift, proceed normally
- 0.1 ≤ PSI < 0.2 → moderate drift, investigate
- PSI ≥ 0.2 → significant drift, trigger retraining workflow

```python
# drift_detection/run_drift_check.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_check(baseline_path: str, window_df: pd.DataFrame) -> dict:
    baseline = pd.read_csv(baseline_path)[["age", "income_k", "tenure_years"]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=window_df)
    results = report.as_dict()

    drift_scores = {}
    for feature in ["age", "income_k", "tenure_years"]:
        psi = results["metrics"][0]["result"]["drift_by_columns"][feature]["drift_score"]
        drift_scores[feature] = psi

    return drift_scores
```

### Drift monitoring DAG (daily)

```python
@dag(schedule_interval="@daily")
def drift_detection():

    @task
    def fetch_recent_inputs() -> pd.DataFrame:
        # Pull last 24 hours of inference inputs from BigQuery
        return bigquery_client.query("""
            SELECT age, income_k, tenure_years
            FROM predictions
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
        """).to_dataframe()

    @task
    def check_drift(window_df: pd.DataFrame):
        scores = run_drift_check("data/dummy_data.csv", window_df)
        for feat, psi in scores.items():
            mlflow.log_metric(f"psi_{feat}", psi)
            if psi >= 0.2:
                alert(f"Significant drift detected in '{feat}' (PSI={psi:.3f}) — retraining recommended")
            elif psi >= 0.1:
                alert(f"Moderate drift in '{feat}' (PSI={psi:.3f}) — monitor closely", severity="warn")
```

### Prediction drift

In addition to input drift, monitor the distribution of predictions themselves:

```python
# Compare daily prediction histogram against a rolling 30-day baseline
# A sudden shift in the mean/variance of predictions is a strong signal
# that either the inputs shifted or the model is behaving unexpectedly.

prediction_mean_today = window_df["prediction"].mean()
prediction_std_today  = window_df["prediction"].std()

baseline_mean = 64_000  # approximate from training
if abs(prediction_mean_today - baseline_mean) / baseline_mean > 0.20:
    alert("Prediction mean shifted > 20% from baseline")
```

---

## Architecture Diagram

```
Production traffic
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Load Balancer / Ingress                         │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────▼──────────┐
       │   Inference Service  │ ──► /metrics ──► Prometheus ──► Grafana
       │   (Flask + Gunicorn) │                                     │
       └──────────┬───────────┘                              Alertmanager
                  │                                               │
           prediction log                               PagerDuty / Slack
                  │
                  ▼
             BigQuery
          (predictions table)
                  │
          ┌───────┴──────────┐
          │                  │
     Actuals join       Drift check
     (nightly DAG)      (daily DAG)
          │                  │
     Performance          Evidently
     metrics              PSI scores
          │                  │
          └──────┬───────────┘
                 │
              MLflow
           (metric history)
```

---

## Trade-offs & Known Limitations

### Trade-offs made in this design

| Decision | Alternative | Reason chosen |
|---|---|---|
| Prometheus + Grafana | Datadog / New Relic | Open-source, no vendor lock-in, works on-prem and in cloud |
| PSI for drift | KL divergence, Wasserstein | PSI is interpretable, well-understood, thresholds are industry standard |
| Daily drift check | Real-time streaming | Sufficient for this use case; streaming adds significant complexity |
| BigQuery for prediction logs | DynamoDB / S3 parquet | Later already uses GCP; BigQuery is cost-effective for analytical queries |
| Async logging from inference service | Synchronous | Adds zero latency to predictions; small risk of log loss under OOM |

### Known limitations

1. **Delayed labels**: Regression ground truth often arrives hours or days later.
   During that window, we can only detect input drift — not performance degradation.
   The monitoring system needs to account for this lag when computing metrics.

2. **Baseline dataset staleness**: The PSI baseline is the original training CSV.
   After retraining, this baseline must be updated. A stale baseline will produce
   misleading PSI scores.

3. **Low-volume cold start**: PSI is unreliable with fewer than ~100 samples per day.
   If request volume is low, switch to a weekly aggregation window.

4. **Single-model assumption**: This design monitors one model version. When running
   A/B tests or canary deployments, metrics must be segmented by `model_version` to
   avoid mixing signals from different models.

5. **No causal attribution**: A drop in model performance could be caused by input drift,
   a data pipeline bug, or a genuine business change. This system detects the signal
   but cannot diagnose the cause automatically — that still requires human investigation.
