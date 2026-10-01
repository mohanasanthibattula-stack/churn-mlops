"""
Unit tests for data ingestion, schema validation, and feature engineering transformations.
Ensures that data cleaning pipelines and feature derivation remain mathematically correct and robust.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import generate_synthetic_churn_data, validate_raw_data
from src.preprocessing import engineer_features, split_data


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Fixture providing a clean minimal raw customer dataframe."""
    return pd.DataFrame({
        "tenure": [1, 12, 24, 60],
        "monthly_charges": [70.0, 80.0, 45.0, 20.0],
        "total_charges": [70.0, 960.0, 1080.0, 1200.0],
        "contract_type": ["Month-to-month", "Month-to-month", "One year", "Two year"],
        "internet_service": ["Fiber optic", "Fiber optic", "DSL", "No"],
        "payment_method": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "tech_support": ["No", "Yes", "Yes", "No internet service"],
        "churn": [1, 1, 0, 0],
    })


def test_synthetic_data_generation_shape_and_columns():
    """Validates that synthetic data generation produces expected volume and mandatory columns."""
    n_rows = 500
    df = generate_synthetic_churn_data(n_samples=n_rows, random_state=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_rows

    expected_columns = {
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "payment_method",
        "tech_support",
        "churn",
    }
    assert expected_columns.issubset(set(df.columns))

    # Verify target is binary and has both classes
    unique_targets = set(df["churn"].unique())
    assert unique_targets == {0, 1}

    # Verify no null or NaN values exist
    assert df.isnull().sum().sum() == 0


def test_validate_raw_data_valid(sample_raw_dataframe: pd.DataFrame):
    """Verifies that a valid dataframe passes schema validation."""
    # Duplicate rows to satisfy minimum size requirement
    large_df = pd.concat([sample_raw_dataframe] * 30, ignore_index=True)
    assert validate_raw_data(large_df) is True


def test_validate_raw_data_missing_column(sample_raw_dataframe: pd.DataFrame):
    """Verifies that missing mandatory columns trigger a ValueError."""
    corrupted_df = sample_raw_dataframe.drop(columns=["monthly_charges"])
    large_corrupted_df = pd.concat([corrupted_df] * 30, ignore_index=True)

    with pytest.raises(ValueError, match="Missing mandatory columns"):
        validate_raw_data(large_corrupted_df)


def test_validate_raw_data_negative_tenure(sample_raw_dataframe: pd.DataFrame):
    """Verifies that negative tenure raises a validation error."""
    corrupted_df = sample_raw_dataframe.copy()
    corrupted_df.loc[0, "tenure"] = -5
    large_corrupted_df = pd.concat([corrupted_df] * 30, ignore_index=True)

    with pytest.raises(ValueError, match="Found negative values in 'tenure'"):
        validate_raw_data(large_corrupted_df)


def test_feature_engineering_adds_expected_columns(sample_raw_dataframe: pd.DataFrame):
    """Validates that feature engineering produces all derived columns."""
    df_featured = engineer_features(sample_raw_dataframe)

    expected_new_cols = {"tenure_group", "avg_monthly_spend", "charge_ratio"}
    assert expected_new_cols.issubset(set(df_featured.columns))


def test_feature_engineering_zero_tenure_safety():
    """Verifies that tenure=0 does not cause division by zero or infinite values."""
    zero_tenure_df = pd.DataFrame({
        "tenure": [0],
        "monthly_charges": [50.0],
        "total_charges": [50.0],
    })

    result = engineer_features(zero_tenure_df)
    avg_spend = result.loc[0, "avg_monthly_spend"]

    assert not np.isnan(avg_spend)
    assert not np.isinf(avg_spend)
    assert avg_spend == 50.0  # 50.0 / (0 + 1)


def test_feature_engineering_tenure_buckets():
    """Verifies that tenure values are mapped to appropriate loyalty brackets."""
    bracket_df = pd.DataFrame({
        "tenure": [6, 18, 36, 65],
        "monthly_charges": [50.0, 50.0, 50.0, 50.0],
        "total_charges": [300.0, 900.0, 1800.0, 3250.0],
    })

    result = engineer_features(bracket_df)
    groups = result["tenure_group"].tolist()

    assert groups == ["0-12m", "13-24m", "25-48m", "49-72m"]


def test_split_data_proportions_and_stratification():
    """Verifies stratified train/test split maintains exact proportions."""
    df = generate_synthetic_churn_data(n_samples=1000, random_state=42)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25, random_state=42)

    assert len(X_train) == 750
    assert len(X_test) == 250
    assert len(y_train) == 750
    assert len(y_test) == 250

    # Verify stratified balance in target classes
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.02
