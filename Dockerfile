# ==============================================================================
# STAGE 1: Build & Dependency Wheel Generation
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Set build environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal compilers required to build C/C++ extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy production requirements only (excludes mlflow, matplotlib, pytest)
COPY requirements.txt .

# Build wheels to isolate build dependencies from runtime
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ==============================================================================
# STAGE 2: Ultra-Lean Production Runtime (<350MB, Non-Root, 512MB RAM Capable)
# ==============================================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Configure lean execution environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install runtime dependencies:
# - libgomp1: OpenMP runtime needed for LightGBM/XGBoost CPU threading
# - curl: Lightweight tool for container healthcheck probes
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user for strict container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -d /app appuser

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source code and trained model artifact with user ownership
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup models/ /app/models/

# Switch to non-root user
USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Health check probe for Render / Hugging Face Spaces / Kubernetes
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn with 1 worker (crucial for staying inside 512MB RAM & 0.1 CPU quota)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
