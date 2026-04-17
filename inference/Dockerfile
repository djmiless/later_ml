# ---------------------------------------------------------------------------
# Multi-stage build — keeps the final image lean
# ---------------------------------------------------------------------------

# Stage 1: build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY inference/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/sh appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
# Build context is the project root, so paths are relative to it.
COPY inference/app.py .

# Copy the trained model artifact.
# NOTE: For production, pull from S3/GCS at startup via MODEL_PATH env var.
# Never bake cloud credentials into the image.
COPY models/regression_model.joblib /app/models/regression_model.joblib

# Runtime configuration
ENV MODEL_PATH=/app/models/regression_model.joblib \
    MODEL_VERSION=1 \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Docker-native health check — Kubernetes reads this status.
# If the model fails to load, /health returns 503 and K8s stops routing traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

USER appuser

# Gunicorn for production — 2 workers, 4 threads each
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", \
     "--timeout", "60", "--access-logfile", "-", "app:app"]
