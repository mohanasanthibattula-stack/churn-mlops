# ==============================================================================
# STAGE 1: Build, Compilation & Binary Stripping
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install compilers and binutils for stripping debug symbols
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ binutils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into an isolated prefix directory
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Strip debug symbols and delete test suites to minimize layer size
RUN find /install -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -name "*.pyc" -delete && \
    find /install -name "*.pyo" -delete


# ==============================================================================
# STAGE 2: Ultra-Lean Production Runtime (<500MB, Non-Root, 512MB RAM Capable)
# ==============================================================================
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install runtime dependencies:
# - libgomp1: OpenMP runtime needed for LightGBM/Scikit-Learn
# - curl: Health check probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create dedicated non-root application user for container security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -d /app appuser

# Copy stripped, pre-compiled packages directly from builder stage
COPY --from=builder /install /usr/local

# Copy application source code and trained model artifact
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup models/ /app/models/

USER appuser
EXPOSE 8000

# Health check probe for Render / Hugging Face Spaces / Kubernetes
HEALTHCHECK --interval=20s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn with 1 worker (essential for 512MB RAM free tier limit)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
