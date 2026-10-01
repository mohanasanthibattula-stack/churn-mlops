"""
Inference module for Customer Churn Prediction.
Loads the serialized production pipeline, accepts raw customer records,
executes automated feature engineering, and generates churn predictions with probability scores.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.preprocessing import engineer_features

# Setup module-level logger
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Singleton-style predictor wrapper that caches the trained model pipeline in memory
    and exposes high-level single-record and batch prediction methods.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initializes the predictor with the trained model pipeline.

        Args:
            model_path (Optional[Path]): Path to serialized model pipeline.
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Any:
        """Loads model pipeline artifact from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run 'python -m src.train' to train a model."
            )
        logger.info(f"Loading churn model pipeline from: {self.model_path}")
        return joblib.load(self.model_path)

    def predict_batch(self, raw_records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Executes predictions on a batch of customer records.

        Args:
            raw_records (Union[pd.DataFrame, List[Dict[str, Any]]]): Raw customer data without derived features.

        Returns:
            List[Dict[str, Any]]: List of prediction result dictionaries with churn probability and risk label.
        """
        if isinstance(raw_records, list):
            df = pd.DataFrame(raw_records)
        else:
            df = raw_records.copy()

        # Step 1: Run feature engineering on raw inference data
        df_featured = engineer_features(df)

        # Step 2: Generate probabilities and binary predictions through pipeline
        probabilities = self.pipeline.predict_proba(df_featured)[:, 1]
        binary_preds = (probabilities >= 0.5).astype(int)

        results = []
        for prob, pred in zip(probabilities, binary_preds):
            prob_float = float(np.round(prob, 4))
            results.append({
                "churn_prediction": int(pred),
                "churn_probability": prob_float,
                "risk_level": "High" if prob_float >= 0.65 else ("Medium" if prob_float >= 0.35 else "Low"),
                "churn_label": "Will Churn" if pred == 1 else "Will Retain",
            })

        return results

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes prediction on a single customer record.

        Args:
            record (Dict[str, Any]): Dictionary of customer attributes.

        Returns:
            Dict[str, Any]: Prediction result with probability and risk category.
        """
        results = self.predict_batch([record])
        return results[0]


# Helper instance for quick programmatic imports
_default_predictor: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """Provides a cached or newly initialized ChurnPredictor instance."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ChurnPredictor()
    return _default_predictor


if __name__ == "__main__":
    sample_customer = {
        "tenure": 2,
        "monthly_charges": 89.5,
        "total_charges": 179.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "tech_support": "No",
    }
    predictor = get_predictor()
    prediction = predictor.predict_single(sample_customer)
    print("\nInference Output for Sample Customer:")
    for k, v in prediction.items():
        print(f"  {k}: {v}")
