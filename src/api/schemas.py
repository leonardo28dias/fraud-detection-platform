from pydantic import BaseModel, Field
from src.data.validation import TransactionSchema

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    applied_threshold: float