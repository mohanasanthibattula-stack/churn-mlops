"""
Model evaluation module for Customer Churn Prediction.
Loads the serialized production model from disk, evaluates performance against
the held-out test split, and logs a comprehensive classification report and confusion matrix.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import settings

# Setup module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
) -> str:
    """
    Evaluates the saved model against the held-out test dataset.

    Args:
        model_path (Optional[Path]): Filepath to serialized model pipeline (.pkl).
        test_data_path (Optional[Path]): Filepath to processed test dataset (.csv).

    Returns:
        str: Formatted classification report string.
    """
    target_model_path = model_path or settings.MODEL_PATH
    target_test_path = test_data_path or settings.TEST_DATA_PATH

    logger.info("--------------------------------------------------")
    logger.info("Starting Model Offline Evaluation")
    logger.info("--------------------------------------------------")

    if not target_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {target_model_path}. Please run 'python -m src.train' first."
        )

    if not target_test_path.exists():
        raise FileNotFoundError(
            f"Test data split not found at {target_test_path}. Please run 'python -m src.train' first."
        )

    logger.info(f"Loading production model from: {target_model_path}")
    pipeline = joblib.load(target_model_path)

    logger.info(f"Loading test split from: {target_test_path}")
    test_df = pd.read_csv(target_test_path)

    if settings.TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{settings.TARGET_COLUMN}' missing from test dataset.")

    X_test = test_df.drop(columns=[settings.TARGET_COLUMN])
    y_test = test_df[settings.TARGET_COLUMN]

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        digits=4,
    )
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("           CUSTOMER CHURN CLASSIFICATION REPORT           ")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print("Confusion Matrix:")
    print(f"  True Negative  (Retained correctly): {cm[0, 0]}")
    print(f"  False Positive (False Alarm):        {cm[0, 1]}")
    print(f"  False Negative (Missed Churn):       {cm[1, 0]}")
    print(f"  True Positive  (Churn detected):     {cm[1, 1]}")
    print("=" * 60 + "\n")

    # If metrics.json exists, verify consistency
    if settings.METRICS_PATH.exists():
        with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        logger.info(f"Logged Training Pipeline Metrics: {saved_metrics}")

    return report


if __name__ == "__main__":
    evaluate_model()
