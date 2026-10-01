"""
Data drift detection module using Evidently AI.
Compares baseline reference training data against live prediction logs recorded
in production (monitoring/prediction_logs.csv) to detect statistical distribution shifts.
Outputs an interactive standalone HTML dashboard for MLOps auditability.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_data_path: Optional[Path] = None,
    current_data_path: Optional[Path] = None,
    output_html_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generates a data drift report comparing reference training features with production inference inputs.

    Args:
        reference_data_path (Optional[Path]): Baseline training split (data/processed/train.csv).
        current_data_path (Optional[Path]): Live prediction log (monitoring/prediction_logs.csv).
        output_html_path (Optional[Path]): File destination for interactive HTML report.

    Returns:
        Dict[str, Any]: Summary dictionary containing drift status and metrics.
    """
    ref_path = reference_data_path or settings.TRAIN_DATA_PATH
    curr_path = current_data_path or (settings.BASE_DIR / "monitoring" / "prediction_logs.csv")
    out_path = output_html_path or (settings.BASE_DIR / "monitoring" / "drift_report.html")

    logger.info("Initializing Evidently AI Data Drift Detection Analysis...")

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference training data not found at {ref_path}. Run training first.")

    # Load baseline reference training data
    ref_df = pd.read_csv(ref_path)
    columns_to_evaluate = [
        col for col in settings.RAW_NUMERICAL_FEATURES + settings.RAW_CATEGORICAL_FEATURES
        if col in ref_df.columns
    ]

    # Load or simulate current inference logs
    if curr_path.exists() and curr_path.stat().st_size > 100:
        logger.info(f"Loading live production inference telemetry from {curr_path}...")
        curr_df = pd.read_csv(curr_path)
    else:
        logger.warning(
            f"Production telemetry at {curr_path} is empty or not yet generated. "
            "Simulating a sample drifted inference cohort for audit report demonstration..."
        )
        # Simulate slight distribution drift for demonstration (e.g., tenure shifted lower, monthly charges higher)
        rng = np.random.default_rng(seed=99)
        curr_df = ref_df.sample(n=min(300, len(ref_df)), random_state=42).copy()
        curr_df["tenure"] = np.clip(curr_df["tenure"] - rng.integers(5, 15, size=len(curr_df)), 0, 72)
        curr_df["monthly_charges"] = np.round(curr_df["monthly_charges"] * 1.15, 2)

    # Filter to identical evaluated feature columns
    ref_eval = ref_df[columns_to_evaluate].copy()
    curr_eval = curr_df[columns_to_evaluate].copy()

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_eval, current_data=curr_eval)

        # Save interactive HTML dashboard
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        logger.info(f"Interactive Evidently Data Drift Report successfully saved to: {out_path}")

        # Extract summary metrics as JSON
        report_dict = report.as_dict()
        drift_metrics = report_dict["metrics"][0]["result"]
        summary = {
            "dataset_drift_detected": drift_metrics.get("dataset_drift", False),
            "drift_share": drift_metrics.get("drift_share", 0.0),
            "number_of_drifted_columns": drift_metrics.get("number_of_drifted_columns", 0),
            "report_html_path": str(out_path),
        }
        logger.info(f"Drift Summary: {summary}")
        return summary

    except ImportError:
        logger.warning(
            "Evidently AI is not installed in the current environment. "
            "Install requirements-dev.txt to render HTML reports."
        )
        # Compute baseline statistical summary via Kolmogorov-Smirnov test fallback
        from scipy.stats import ks_2samp

        drift_detected_cols = []
        for col in settings.RAW_NUMERICAL_FEATURES:
            stat, p_val = ks_2samp(ref_eval[col], curr_eval[col])
            if p_val < 0.05:
                drift_detected_cols.append(col)

        summary = {
            "dataset_drift_detected": len(drift_detected_cols) > 0,
            "drifted_features": drift_detected_cols,
            "note": "Computed using SciPy KS-test fallback.",
        }
        logger.info(f"Statistical KS-test Summary: {summary}")
        return summary


if __name__ == "__main__":
    generate_drift_report()
