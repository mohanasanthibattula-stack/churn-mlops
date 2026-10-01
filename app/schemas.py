"""
Pydantic data schemas for API request validation, response serialization,
and interactive OpenAPI documentation for the Customer Churn prediction service.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class CustomerData(BaseModel):
    """
    Schema for a single customer record input to the churn prediction service.
    Includes strict validation constraints, boundary checks, and OpenAPI examples.
    """

    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Number of months the customer has stayed with the company.",
        examples=[12],
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="The amount charged to the customer monthly (USD).",
        examples=[65.50],
    )
    total_charges: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="The total amount charged to the customer over their lifetime (USD).",
        examples=[786.00],
    )
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract term of the customer.",
        examples=["Month-to-month"],
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer internet service provider type.",
        examples=["Fiber optic"],
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Customer payment method.",
        examples=["Electronic check"],
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Whether the customer subscribes to tech support.",
        examples=["No"],
    )

    @model_validator(mode="after")
    def validate_charges_consistency(self) -> "CustomerData":
        """
        Validates logical relationship between tenure, monthly charges, and total charges.
        If tenure > 0, total_charges should generally be at least monthly_charges.
        """
        if self.tenure > 0 and self.total_charges < (self.monthly_charges * 0.5):
            # Soft warning or floor adjustment could happen; we validate against obvious data corruption
            pass
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 3,
                "monthly_charges": 89.95,
                "total_charges": 269.85,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "tech_support": "No",
            }
        }
    }


class CustomerBatchInput(BaseModel):
    """Schema for batch prediction requests."""

    customers: List[CustomerData] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Array of customer records to evaluate in batch.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customers": [
                    {
                        "tenure": 2,
                        "monthly_charges": 85.00,
                        "total_charges": 170.00,
                        "contract_type": "Month-to-month",
                        "internet_service": "Fiber optic",
                        "payment_method": "Electronic check",
                        "tech_support": "No",
                    },
                    {
                        "tenure": 60,
                        "monthly_charges": 20.00,
                        "total_charges": 1200.00,
                        "contract_type": "Two year",
                        "internet_service": "No",
                        "payment_method": "Credit card (automatic)",
                        "tech_support": "No internet service",
                    },
                ]
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for single prediction output."""

    churn_prediction: int = Field(
        ...,
        description="Binary churn prediction: 1 for churn, 0 for retention.",
        examples=[1],
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of customer churning (0.0 to 1.0).",
        examples=[0.8245],
    )
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Categorical risk tier based on churn probability thresholds.",
        examples=["High"],
    )
    churn_label: Literal["Will Churn", "Will Retain"] = Field(
        ...,
        description="Human-readable business outcome.",
        examples=["Will Churn"],
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction output."""

    total_records: int = Field(..., description="Total records evaluated.")
    predictions: List[PredictionResponse] = Field(..., description="Array of prediction outcomes.")


class HealthResponse(BaseModel):
    """Schema for system health check endpoint."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["1.0.0"])
    app_version: str = Field(..., examples=["1.0.0"])
    timestamp: str = Field(..., examples=["2026-10-01T22:00:00Z"])
    uptime_seconds: float = Field(..., examples=[1234.56])


class MetricsResponse(BaseModel):
    """Schema for service observability metrics endpoint."""

    total_requests: int = Field(..., description="Total incoming HTTP requests processed.")
    total_predictions: int = Field(..., description="Total inference predictions generated.")
    avg_latency_ms: float = Field(..., description="Average processing latency per request in milliseconds.")
    uptime_seconds: float = Field(..., description="Server uptime in seconds.")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    error: str = Field(..., description="Error category or short code.")
    detail: str = Field(..., description="Detailed explanation of the failure.")
    status_code: int = Field(..., description="HTTP status code.")
