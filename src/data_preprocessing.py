"""
Data preprocessing module for Customer Churn Prediction.
Builds the numerical and categorical transformation pipelines using scikit-learn ColumnTransformer,
handles missing values, performs stratified splitting, and serializes preprocessing artifacts.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.feature_engineering import engineer_features

# Setup module-level logger
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for preprocessing numerical
    and categorical features.

    Numerical Pipeline:
        - Median Imputation (guards against empty total charges)
        - Standard Scaling (zero mean, unit variance)

    Categorical Pipeline:
        - Frequent Value Imputation
        - One-Hot Encoding (ignoring unknown categories at inference time)

    Returns:
        ColumnTransformer: Configured preprocessor object.
    """
    logger.info("Configuring preprocessing ColumnTransformer pipelines...")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, settings.ENGINEERED_NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, settings.ENGINEERED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = settings.TEST_SIZE,
    random_state: int = settings.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into stratified training and testing sets.

    Args:
        df (pd.DataFrame): Dataset containing features and target column.
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test.
    """
    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}...")

    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{settings.TARGET_COLUMN}' not found in dataframe.")

    X = df.drop(columns=[settings.TARGET_COLUMN])
    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Data split successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: Union[str, Path] = settings.TRAIN_DATA_PATH,
    test_path: Union[str, Path] = settings.TEST_DATA_PATH,
) -> None:
    """
    Saves processed train and test splits to CSV files for reproducibility.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        train_path (Path): Destination path for training split.
        test_path (Path): Destination path for test split.
    """
    train_df = X_train.copy()
    train_df[settings.TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[settings.TARGET_COLUMN] = y_test

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Train dataset saved to: {train_path}")
    logger.info(f"Test dataset saved to: {test_path}")


def save_preprocessor(preprocessor: ColumnTransformer, path: Union[str, Path]) -> None:
    """
    Persists fitted ColumnTransformer to disk using joblib.

    Args:
        preprocessor (ColumnTransformer): Fitted preprocessor object.
        path (Path): File destination for serialization.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, target)
    logger.info(f"Preprocessor saved successfully to: {target}")


def load_preprocessor(path: Union[str, Path]) -> ColumnTransformer:
    """
    Loads serialized ColumnTransformer from disk.

    Args:
        path (Path): Filepath of saved preprocessor.

    Returns:
        ColumnTransformer: Loaded preprocessor.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {target}")
    logger.info(f"Loading preprocessor from: {target}")
    return joblib.load(target)
