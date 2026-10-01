"""
Efficiency-tuned model loader and in-memory cache manager.
Features:
- Singleton lifecycle: model is loaded exactly once into RAM at server boot.
- Uses shared src.preprocessing.engineer_features (DRY).
- LRU caching: frequent prediction payloads return in < 0.1ms without pipeline re-evaluation.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.preprocessing import engineer_features

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager holding the in-memory ML pipeline with LRU prediction caching.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path: Path = model_path or settings.MODEL_PATH
        self._pipeline: Optional[Any] = None

    def load_model(self) -> None:
        """Loads serialized pipeline artifact once into memory."""
        if not self.model_path.exists():
            error_msg = f"Model artifact not found at {self.model_path}. Run training first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading lean model pipeline from {self.model_path} into memory...")
        self._pipeline = joblib.load(self.model_path)
        logger.info("Model pipeline ready in RAM.")

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is ready in memory."""
        return self._pipeline is not None

    def predict_single_cached(
        self,
        tenure: int,
        monthly_charges: float,
        total_charges: float,
        contract_type: str,
        internet_service: str,
        payment_method: str,
        tech_support: str,
    ) -> Dict[str, Any]:
        """
        LRU-cached single prediction method for sub-millisecond responses on repeated inputs.
        """
        return self._cached_predict_tuple((
            tenure,
            monthly_charges,
            total_charges,
            contract_type,
            internet_service,
            payment_method,
            tech_support,
        ))

    @lru_cache(maxsize=1024)
    def _cached_predict_tuple(self, features: Tuple[Any, ...]) -> Dict[str, Any]:
        """Internal tuple-based LRU prediction cache."""
        record = {
            "tenure": features[0],
            "monthly_charges": features[1],
            "total_charges": features[2],
            "contract_type": features[3],
            "internet_service": features[4],
            "payment_method": features[5],
            "tech_support": features[6],
        }
        return self.predict_batch([record])[0]

    def predict_batch(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        High-speed batch prediction using vectorized preprocessing.
        """
        if not self.is_loaded:
            raise RuntimeError("Model pipeline is not loaded. Call load_model() first.")

        df = pd.DataFrame(raw_data)
        df_featured = engineer_features(df)

        probabilities = self._pipeline.predict_proba(df_featured)[:, 1]
        binary_preds = (probabilities >= 0.5).astype(int)

        results = []
        for prob, pred in zip(probabilities, binary_preds):
            p = float(np.round(prob, 4))
            tier = "High" if p >= 0.65 else ("Medium" if p >= 0.35 else "Low")
            results.append({
                "churn_prediction": int(pred),
                "churn_probability": p,
                "risk_level": tier,
                "churn_label": "Will Churn" if pred == 1 else "Will Retain",
            })
        return results


# Global singleton instance
model_manager = ModelManager()
