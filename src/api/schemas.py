from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    transaction_id: str = Field(
        ..., description="Unique identifier for this prediction"
    )
    is_fraud: bool = Field(
        ..., description="Whether the transaction is classified as fraud"
    )
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model fraud probability"
    )
    applied_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Decision threshold used"
    )
