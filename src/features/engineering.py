import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-of-day features from elapsed seconds to capture daily cyclical patterns
    without looking ahead in time.
    """
    def __init__(self, time_col: str = 'Time'):
        self.time_col = time_col
        self.seconds_in_day = 86400

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        hours = (X_out[self.time_col] % self.seconds_in_day) / 3600
        X_out['Hour_Sin'] = np.sin(2 * np.pi * hours / 24)
        X_out['Hour_Cos'] = np.cos(2 * np.pi * hours / 24)
        X_out = X_out.drop(columns=[self.time_col])
        return X_out

class LogAmountTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to Amount to handle extreme right skewness.
    """
    def __init__(self, amount_col: str = 'Amount'):
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out['Log_Amount'] = np.log1p(X_out[self.amount_col])
        X_out = X_out.drop(columns=[self.amount_col])
        return X_out