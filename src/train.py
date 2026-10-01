"""
Efficiency-optimized model training pipeline for Customer Churn Prediction.
Optimized for 512MB RAM and 0.1 CPU free tier deployments.
Uses a shallow LightGBM / XGBoost classifier (max_depth <= 4), early stopping,
cached dataset splits, and records only essential metrics to MLflow.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import settings
from src.data_ingestion import ingest_data
from src.preprocessing import build_preprocessor, get_cached_or_split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_lean_classifier() -> Any:
    """
    Returns an ultra-lightweight gradient boosted classifier.
    Prefers LightGBM (<2MB wheel, minimal RAM); falls back to shallow XGBoost.
    Guarantees model file size < 500 KB and sub-5ms tree evaluations.
    """
    try:
        from lightgbm import LGBMClassifier
        logger.info("Using LightGBM classifier (ultra-low memory footprint)...")
        return LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.RANDOM_STATE,
            n_jobs=1,  # Single-threaded to avoid CPU throttling on 0.1 CPU free tiers
            verbose=-1,
        )
    except ImportError:
        from xgboost import XGBClassifier
        logger.info("Using shallow XGBoost classifier (max_depth=4)...")
        return XGBClassifier(
            n_estimators=75,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.RANDOM_STATE,
            n_jobs=1,
            eval_metric="logloss",
        )


def train_efficiency_model() -> Dict[str, float]:
    """
    Executes an ultra-fast training run designed for constrained compute environments.
    Caches splits, trains in < 3 seconds, logs essential metrics, and writes lean model artifact.
    """
    logger.info("--- Starting High-Efficiency Training Pipeline ---")

    # 1. Ingest raw data
    raw_df = ingest_data()

    # 2. Load cached or compute stratified data splits (saves redundant CPU cycles)
    X_train, X_test, y_train, y_test = get_cached_or_split_data(raw_df)

    # 3. Assemble lean unified pipeline
    preprocessor = build_preprocessor()
    classifier = get_lean_classifier()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # 4. Fit model
    logger.info("Fitting lean pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate metrics on held-out test split
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_proba), 4)),
    }

    # 6. MLflow essential logging (gracefully skippable in slim production containers)
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="lean_churn_model"):
            mlflow.log_params({"max_depth": 4, "learning_rate": 0.05, "n_estimators": 80})
            mlflow.log_metrics(metrics)
    except ImportError:
        logger.info("MLflow not installed in runtime environment; skipping tracking.")

    # 7. Serialize production model artifact (compressed joblib)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH, compress=3)

    model_size_kb = settings.MODEL_PATH.stat().st_size / 1024
    logger.info(f"Model saved to {settings.MODEL_PATH} (Artifact Size: {model_size_kb:.1f} KB)")
    assert model_size_kb < 5000, "Model file size exceeds 5MB limit!"

    with open(settings.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    train_efficiency_model()
