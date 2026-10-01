"""
Data ingestion module for Customer Churn Prediction.
Loads raw data from disk or generates a realistic synthetic Telco churn dataset (>= 5,000 rows)
if the raw source file is not found, followed by schema and statistical validation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import settings

# Setup module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_churn_data(
    n_samples: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer churn dataset mimicking the Kaggle Telco Churn dataset.

    Args:
        n_samples (int): Number of customer records to generate (default: 6,000).
        random_state (int): Seed for reproducible random generation.

    Returns:
        pd.DataFrame: Synthetic dataset with realistic feature distributions and relationships.
    """
    logger.info(f"Generating synthetic churn dataset with {n_samples} records...")
    rng = np.random.default_rng(seed=random_state)

    # 1. Tenure in months: Uniform distribution between 1 and 72 months
    tenure = rng.integers(1, 73, size=n_samples)

    # 2. Contract Type: Month-to-month, One year, Two year
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract_type = rng.choice(contract_choices, size=n_samples, p=contract_probs)

    # 3. Internet Service: DSL, Fiber optic, No
    internet_choices = ["DSL", "Fiber optic", "No"]
    internet_probs = [0.44, 0.44, 0.12]
    internet_service = rng.choice(internet_choices, size=n_samples, p=internet_probs)

    # 4. Tech Support: Dependent on internet service
    tech_support = []
    for net in internet_service:
        if net == "No":
            tech_support.append("No internet service")
        else:
            tech_support.append(rng.choice(["Yes", "No"], p=[0.35, 0.65]))
    tech_support = np.array(tech_support)

    # 5. Payment Method
    payment_choices = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    payment_probs = [0.35, 0.22, 0.22, 0.21]
    payment_method = rng.choice(payment_choices, size=n_samples, p=payment_probs)

    # 6. Monthly Charges: Correlated with Internet Service tier
    monthly_charges = np.zeros(n_samples)
    for i, net in enumerate(internet_service):
        if net == "Fiber optic":
            monthly_charges[i] = rng.uniform(70.0, 118.0)
        elif net == "DSL":
            monthly_charges[i] = rng.uniform(35.0, 85.0)
        else:
            monthly_charges[i] = rng.uniform(18.5, 25.5)
    monthly_charges = np.round(monthly_charges, 2)

    # 7. Total Charges: Proportional to tenure * monthly charges with slight realistic variance
    variance_noise = rng.normal(1.0, 0.05, size=n_samples)
    total_charges = np.round(np.maximum(monthly_charges, tenure * monthly_charges * variance_noise), 2)

    # 8. Churn Target generation using realistic log-odds calculation
    # Factors increasing churn: Month-to-month contract, High monthly charges, Electronic check, No tech support
    # Factors decreasing churn: Long tenure, Two-year contract, Has tech support
    log_odds = -1.2  # Baseline intercept (~23% churn baseline)
    log_odds += (contract_type == "Month-to-month") * 1.3
    log_odds -= (contract_type == "Two year") * 1.5
    log_odds -= (contract_type == "One year") * 0.7
    log_odds += (internet_service == "Fiber optic") * 0.6
    log_odds += (payment_method == "Electronic check") * 0.8
    log_odds += (tech_support == "No") * 0.5
    log_odds -= (tenure / 72.0) * 1.8
    log_odds += ((monthly_charges - 65.0) / 30.0) * 0.4

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support,
        "churn": churn,
    })

    logger.info(f"Synthetic dataset generated successfully with churn rate: {df['churn'].mean():.2%}")
    return df


def validate_raw_data(df: pd.DataFrame) -> bool:
    """
    Validates the structure and basic integrity of the ingested dataset.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Raises:
        ValueError: If mandatory columns are missing or data types are invalid.

    Returns:
        bool: True if validation passes.
    """
    required_columns = set(
        settings.RAW_NUMERICAL_FEATURES + settings.RAW_CATEGORICAL_FEATURES + [settings.TARGET_COLUMN]
    )
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        error_msg = f"Validation failed: Missing mandatory columns: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(df) < 100:
        error_msg = f"Validation failed: Dataset is too small ({len(df)} rows)."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if (df["tenure"] < 0).any():
        raise ValueError("Validation failed: Found negative values in 'tenure'.")

    if (df["monthly_charges"] < 0).any() or (df["total_charges"] < 0).any():
        raise ValueError("Validation failed: Found negative values in charge columns.")

    logger.info("Raw data validation checks passed successfully.")
    return True


def ingest_data(
    file_path: Optional[Path] = None,
    save_if_generated: bool = True,
) -> pd.DataFrame:
    """
    Ingests churn dataset from file if available, or generates realistic synthetic data.

    Args:
        file_path (Optional[Path]): Path to the raw CSV file. Defaults to settings.RAW_DATA_PATH.
        save_if_generated (bool): Whether to persist the generated dataset to disk.

    Returns:
        pd.DataFrame: Validated raw customer churn dataset.
    """
    target_path = file_path or settings.RAW_DATA_PATH

    if target_path.exists():
        logger.info(f"Loading raw data from existing file: {target_path}")
        try:
            df = pd.read_csv(target_path)
        except Exception as e:
            logger.error(f"Failed to read existing raw data at {target_path}: {e}")
            raise
    else:
        logger.warning(f"Raw data file not found at {target_path}. Generating synthetic dataset...")
        df = generate_synthetic_churn_data(n_samples=6000, random_state=settings.RANDOM_STATE)

        if save_if_generated:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target_path, index=False)
            logger.info(f"Saved generated synthetic dataset to {target_path}")

    validate_raw_data(df)
    return df


if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
    print("\nDataset info:")
    print(df.info())
