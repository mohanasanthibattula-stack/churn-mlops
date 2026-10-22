"""
Central configuration module for the Customer Churn MLOps pipeline.
Handles file paths, dataset schema, model hyperparameters, and environment variables.
"""

import os
from pathlib import Path
from typing import List

# Base directory: root of the churn-mlops repository
BASE_DIR: Path = Path(__file__).resolve().parent.parent

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class PipelineSettings(BaseSettings):
        """Application and pipeline configuration using pydantic-settings."""
        model_config = SettingsConfigDict(
            env_prefix="CHURN_",
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # Project Metadata
        PROJECT_NAME: str = "Customer Churn Prediction"
        VERSION: str = "1.0.0"
        DEBUG: bool = False

        # Directory Paths
        DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
        DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
        MODELS_DIR: Path = BASE_DIR / "models"
        MLRUNS_DIR: Path = BASE_DIR / "mlruns"

        # File Paths
        RAW_DATA_PATH: Path = DATA_RAW_DIR / "churn_data.csv"
        TRAIN_DATA_PATH: Path = DATA_PROCESSED_DIR / "train.csv"
        TEST_DATA_PATH: Path = DATA_PROCESSED_DIR / "test.csv"
        MODEL_PATH: Path = MODELS_DIR / "model.pkl"
        METRICS_PATH: Path = MODELS_DIR / "metrics.json"

        # Dataset Column Definitions
        TARGET_COLUMN: str = "churn"
        RAW_NUMERICAL_FEATURES: List[str] = [
            "tenure",
            "monthly_charges",
            "total_charges",
        ]
        RAW_CATEGORICAL_FEATURES: List[str] = [
            "contract_type",
            "internet_service",
            "payment_method",
            "tech_support",
        ]

        # Features after Feature Engineering
        ENGINEERED_NUMERICAL_FEATURES: List[str] = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "avg_monthly_spend",
            "charge_ratio",
        ]
        ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
            "contract_type",
            "internet_service",
            "payment_method",
            "tech_support",
            "tenure_group",
        ]

        # Model Hyperparameters
        RANDOM_STATE: int = 42
        TEST_SIZE: float = 0.20
        N_ESTIMATORS: int = 150
        MAX_DEPTH: int = 4
        LEARNING_RATE: float = 0.05
        SUBSAMPLE: float = 0.80
        COLSAMPLE_BYTREE: float = 0.80

        # MLflow Tracking
        MLFLOW_EXPERIMENT_NAME: str = "churn-prediction-xgboost"
        MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns"))

        # API Serving
        API_HOST: str = "0.0.0.0"
        API_PORT: int = 8000

    settings = PipelineSettings()

except ImportError:
    # Graceful fallback to pure Python configuration if pydantic-settings is not installed yet
    class PipelineSettings:  # type: ignore[no-redef]
        """Fallback configuration using standard os.getenv and pathlib."""
        PROJECT_NAME: str = os.getenv("CHURN_PROJECT_NAME", "Customer Churn Prediction")
        VERSION: str = os.getenv("CHURN_VERSION", "1.0.0")
        DEBUG: bool = os.getenv("CHURN_DEBUG", "False").lower() in ("true", "1", "yes")

        DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
        DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
        MODELS_DIR: Path = BASE_DIR / "models"
        MLRUNS_DIR: Path = BASE_DIR / "mlruns"

        RAW_DATA_PATH: Path = DATA_RAW_DIR / "churn_data.csv"
        TRAIN_DATA_PATH: Path = DATA_PROCESSED_DIR / "train.csv"
        TEST_DATA_PATH: Path = DATA_PROCESSED_DIR / "test.csv"
        MODEL_PATH: Path = MODELS_DIR / "model.pkl"
        METRICS_PATH: Path = MODELS_DIR / "metrics.json"

        TARGET_COLUMN: str = "churn"
        RAW_NUMERICAL_FEATURES: List[str] = [
            "tenure",
            "monthly_charges",
            "total_charges",
        ]
        RAW_CATEGORICAL_FEATURES: List[str] = [
            "contract_type",
            "internet_service",
            "payment_method",
            "tech_support",
        ]

        ENGINEERED_NUMERICAL_FEATURES: List[str] = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "avg_monthly_spend",
            "charge_ratio",
        ]
        ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
            "contract_type",
            "internet_service",
            "payment_method",
            "tech_support",
            "tenure_group",
        ]

        RANDOM_STATE: int = int(os.getenv("CHURN_RANDOM_STATE", "42"))
        TEST_SIZE: float = float(os.getenv("CHURN_TEST_SIZE", "0.20"))
        N_ESTIMATORS: int = int(os.getenv("CHURN_N_ESTIMATORS", "150"))
        MAX_DEPTH: int = int(os.getenv("CHURN_MAX_DEPTH", "4"))
        LEARNING_RATE: float = float(os.getenv("CHURN_LEARNING_RATE", "0.05"))
        SUBSAMPLE: float = float(os.getenv("CHURN_SUBSAMPLE", "0.80"))
        COLSAMPLE_BYTREE: float = float(os.getenv("CHURN_COLSAMPLE_BYTREE", "0.80"))

        MLFLOW_EXPERIMENT_NAME: str = os.getenv("CHURN_MLFLOW_EXPERIMENT_NAME", "churn-prediction-xgboost")
        MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", str(BASE_DIR / "mlruns"))

        API_HOST: str = os.getenv("CHURN_API_HOST", "0.0.0.0")
        API_PORT: int = int(os.getenv("CHURN_API_PORT", "8000"))

    settings = PipelineSettings()

# Ensure directories exist upon config load (silently ignore permission errors in restricted container environments)
for _dir in (settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR, settings.MODELS_DIR, settings.MLRUNS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

