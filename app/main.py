"""
Production FastAPI application for Customer Churn Prediction Serving.
Optimized for 512MB RAM free tier instances (Render / Hugging Face Spaces):
- Asynchronous lifespan model loading
- LRU caching for frequent prediction payloads (< 1ms response)
- Minimal JSON response footprint
- Ultra-fast Pydantic rejection of invalid payloads
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model_loader import model_manager
from app.schemas import (
    BatchPredictionResponse,
    CustomerBatchInput,
    CustomerData,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from src.config import settings

# Configure lean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("churn-api")

APP_START_TIME: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads model into memory once at container start."""
    logger.info("FastAPI service booting. Initializing in-memory model cache...")
    try:
        model_manager.load_model()
    except FileNotFoundError as e:
        logger.warning(f"Startup warning: {e}. Model will need to be trained prior to inference.")
    yield
    logger.info("FastAPI service stopped.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-efficiency, low-latency churn prediction microservice.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,  # Disabled to save ~2MB memory footprint
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures latency and injects lightweight timing header."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Fast rejection of malformed payloads before touching inference code."""
    errors = exc.errors()
    msg = errors[0].get("msg") if errors else "Invalid payload"
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation Error", "detail": str(msg), "status_code": 400},
    )


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """Root status endpoint returning minimal JSON."""
    return {"service": settings.PROJECT_NAME, "status": "online", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health() -> HealthResponse:
    """Lightweight health check endpoint for container probes."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=settings.VERSION,
        app_version=settings.VERSION,
        timestamp=str(int(time.time())),
        uptime_seconds=round(time.time() - APP_START_TIME, 1),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def predict(customer: CustomerData) -> PredictionResponse:
    """
    Sub-100ms prediction endpoint with LRU caching for repeated payloads.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is initializing. Please retry in a moment.",
        )

    try:
        # Check LRU cache or execute pipeline
        result = model_manager.predict_single_cached(
            tenure=customer.tenure,
            monthly_charges=customer.monthly_charges,
            total_charges=customer.total_charges,
            contract_type=customer.contract_type,
            internet_service=customer.internet_service,
            payment_method=customer.payment_method,
            tech_support=customer.tech_support,
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction execution failed.",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def predict_batch(batch: CustomerBatchInput) -> BatchPredictionResponse:
    """High-throughput vectorized batch prediction endpoint."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is initializing.",
        )

    try:
        raw_items = [c.model_dump() for c in batch.customers]
        results = model_manager.predict_batch(raw_items)
        return BatchPredictionResponse(
            total_records=len(results),
            predictions=[PredictionResponse(**r) for r in results],
        )
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        )
