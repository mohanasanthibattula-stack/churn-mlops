"""
Shared preprocessing module for Customer Churn Prediction.
Used by BOTH model training and real-time API serving (DRY principle).
Uses strictly vectorized NumPy/Pandas operations for sub-10ms feature transformations.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized feature engineering shared identically across train and inference.
    Drops low-importance noise features and keeps only high-impact signals.

    Features generated:
        - tenure_group: Discretized loyalty bracket
        - avg_monthly_spend: Ratio of total charges to tenure
        - charge_ratio: Ratio of current monthly charge to lifetime spend

    Args:
        df (pd.DataFrame): Input dataframe with raw customer records.

    Returns:
        pd.DataFrame: Transformed dataframe with high-signal engineered features.
    """
    data = df.copy()

    # 1. Vectorized tenure discretization (discards raw noise, groups into cohorts)
    bins = [-1, 12, 24, 48, 120]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
    data["tenure_group"] = pd.cut(data["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Vectorized average monthly spend rate
    data["avg_monthly_spend"] = np.round(data["total_charges"] / (data["tenure"] + 1), 2)

    # 3. Vectorized burden ratio
    data["charge_ratio"] = np.round(data["monthly_charges"] / (data["total_charges"] + 1), 4)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an ultra-lean ColumnTransformer pipeline.
    Uses sparse_output=False for fast dense NumPy memory layouts.

    Returns:
        ColumnTransformer: Scaler and One-Hot encoding pipeline.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, settings.ENGINEERED_NUMERICAL_FEATURES),
            ("cat", cat_pipe, settings.ENGINEERED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_cached_or_split_data(
    df: pd.DataFrame,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads cached processed splits from disk to minimize CPU usage on free tiers,
    or generates new stratified splits if cache is absent.

    Args:
        df (pd.DataFrame): Raw ingested dataset.
        force_refresh (bool): When True, regenerates cache.

    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    train_cache = settings.TRAIN_DATA_PATH
    test_cache = settings.TEST_DATA_PATH

    if not force_refresh and train_cache.exists() and test_cache.exists():
        logger.info("Loading preprocessed dataset splits directly from disk cache...")
        train_df = pd.read_csv(train_cache)
        test_df = pd.read_csv(test_cache)
        X_train = train_df.drop(columns=[settings.TARGET_COLUMN])
        y_train = train_df[settings.TARGET_COLUMN]
        X_test = test_df.drop(columns=[settings.TARGET_COLUMN])
        y_test = test_df[settings.TARGET_COLUMN]
        return X_train, X_test, y_train, y_test

    logger.info("Generating and caching new stratified dataset splits...")
    df_featured = engineer_features(df)
    X = df_featured.drop(columns=[settings.TARGET_COLUMN])
    y = df_featured[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )

    # Persist cache to disk
    train_df = X_train.copy()
    train_df[settings.TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[settings.TARGET_COLUMN] = y_test

    train_cache.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_cache, index=False)
    test_df.to_csv(test_cache, index=False)
    logger.info(f"Cached train and test splits to {train_cache.parent}")

    return X_train, X_test, y_train, y_test


def split_data(
    df: pd.DataFrame,
    test_size: float = settings.TEST_SIZE,
    random_state: int = settings.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Standard stratified train/test split without disk caching.
    """
    if "churn" in df.columns and "avg_monthly_spend" not in df.columns:
        df_featured = engineer_features(df)
    else:
        df_featured = df.copy()

    X = df_featured.drop(columns=[settings.TARGET_COLUMN])
    y = df_featured[settings.TARGET_COLUMN]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
