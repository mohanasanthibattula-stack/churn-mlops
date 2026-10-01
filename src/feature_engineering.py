"""
Feature engineering module for Customer Churn Prediction.
Creates derived domain features such as tenure buckets, average monthly spend,
and charge ratios to enhance the predictive power of the model.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Setup module-level logger
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived domain-specific features from raw customer records.

    Features generated:
        - tenure_group: Categorical bucket based on customer loyalty duration.
        - avg_monthly_spend: Ratio of total charges to tenure (spend rate).
        - charge_ratio: Ratio of current monthly charge to lifetime total spend.

    Args:
        df (pd.DataFrame): Raw or ingested customer dataset.

    Returns:
        pd.DataFrame: A new DataFrame with the added engineered feature columns.
    """
    logger.info("Starting feature engineering transformation...")
    data = df.copy()

    # Validate essential columns exist for feature derivation
    required_cols = {"tenure", "monthly_charges", "total_charges"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Cannot engineer features. Missing required columns: {missing}")

    # 1. Tenure Buckets: Discretize tenure into meaningful loyalty brackets
    # 0-12 months (High risk new users), 13-24 months (Developing), 25-48 months (Established), 49+ months (Loyal)
    bins = [-1, 12, 24, 48, 100]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
    data["tenure_group"] = pd.cut(data["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Average Monthly Spend (safely avoiding division by zero for new signups)
    data["avg_monthly_spend"] = np.round(data["total_charges"] / (data["tenure"] + 1), 2)

    # 3. Monthly to Total Charge Ratio (indicates relative recent cost burden)
    data["charge_ratio"] = np.round(data["monthly_charges"] / (data["total_charges"] + 1), 4)

    logger.info(f"Feature engineering completed successfully. Added 3 new features: {['tenure_group', 'avg_monthly_spend', 'charge_ratio']}")
    return data


# Scikit-learn Transformer compatibility wrapper
try:
    from sklearn.base import BaseEstimator, TransformerMixin

    class FeatureEngineer(BaseEstimator, TransformerMixin):
        """
        Scikit-learn compatible transformer that wraps feature engineering logic,
        allowing it to be integrated seamlessly into a unified Pipeline.
        """

        def __init__(self) -> None:
            pass

        def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return engineer_features(X)

except ImportError:
    # Standalone placeholder if scikit-learn is not installed in the active environment
    class FeatureEngineer:  # type: ignore[no-redef]
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return engineer_features(X)


if __name__ == "__main__":
    from src.data_ingestion import ingest_data

    raw_df = ingest_data()
    featured_df = engineer_features(raw_df)
    print("\nSample engineered features:")
    print(featured_df[["tenure", "tenure_group", "monthly_charges", "total_charges", "avg_monthly_spend", "charge_ratio"]].head())
