import logging
from typing import List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TransactionSchema(BaseModel):
    """Pydantic schema for a single credit-card transaction.

    Used both for raw-data validation and as the FastAPI request body.
    V1-V28 are PCA-anonymised features; Time is seconds elapsed since the
    first transaction; Amount is the transaction value in the original currency.
    """

    model_config = ConfigDict(strict=False)

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0.0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw dataframe against the TransactionSchema."""
    records = df.to_dict(orient="records")
    validated: List[TransactionSchema] = [
        TransactionSchema(**record) for record in records
    ]
    return pd.DataFrame([v.model_dump() for v in validated])
