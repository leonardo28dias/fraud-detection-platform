import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour-of-day cyclical features from elapsed seconds.

    Converts the raw Time column (seconds since first transaction) into
    sin/cos features that capture daily patterns without looking ahead.
    The raw Time column is dropped.
    """

    def __init__(self, time_col: str = "Time"):
        self.time_col = time_col
        self.seconds_in_day: int = 86400

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        hours = (X_out[self.time_col] % self.seconds_in_day) / 3600
        X_out["Hour_Sin"] = np.sin(2 * np.pi * hours / 24)
        X_out["Hour_Cos"] = np.cos(2 * np.pi * hours / 24)
        X_out = X_out.drop(columns=[self.time_col])
        return X_out


class LogAmountTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to the Amount column.

    Handles extreme right-skew in transaction amounts.
    The raw Amount column is replaced with Log_Amount.
    """

    def __init__(self, amount_col: str = "Amount"):
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y=None) -> "LogAmountTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out["Log_Amount"] = np.log1p(X_out[self.amount_col])
        X_out = X_out.drop(columns=[self.amount_col])
        return X_out
