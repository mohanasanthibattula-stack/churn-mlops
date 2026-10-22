# ==============================================================================
# STAGE 1: Build & Dependency Wheel Generation
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install compilers to build binary wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels to isolate compilation tools from runtime
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ==============================================================================
# STAGE 2: Production Runtime (<600MB, Non-Root, 512MB RAM Capable)
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

# Install pre-built wheels into standard site-packages and purge unneeded test suites
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels && \
    find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete

# Copy application source code and trained model artifact with user ownership
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
