import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing import List

class TransactionSchema(BaseModel):
    model_config = ConfigDict(strict=False)
    
    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction")
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
    """Validates raw dataframe against Pydantic schema."""
    records = df.to_dict(orient="records")
    validated = [TransactionSchema(**record).model_dump() for record in records]
    return pd.DataFrame(validated)