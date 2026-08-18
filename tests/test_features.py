import pandas as pd
import numpy as np
from src.features.engineering import TemporalFeatureExtractor, LogAmountTransformer

def test_temporal_feature_extractor():
    df = pd.DataFrame({'Time': [0, 43200, 86400]}) # Midnight, Noon, Midnight next day
    extractor = TemporalFeatureExtractor()
    transformed = extractor.fit_transform(df)
    
    assert 'Time' not in transformed.columns
    assert 'Hour_Sin' in transformed.columns
    assert 'Hour_Cos' in transformed.columns
    
    # Midnight sin should be ~0, cos ~1
    np.testing.assert_almost_equal(transformed['Hour_Sin'].iloc[0], 0, decimal=2)
    np.testing.assert_almost_equal(transformed['Hour_Cos'].iloc[0], 1, decimal=2)

def test_log_amount_transformer():
    df = pd.DataFrame({'Amount': [0, 99]})
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(df)
    
    assert 'Amount' not in transformed.columns
    assert 'Log_Amount' in transformed.columns
    assert transformed['Log_Amount'].iloc[0] == 0.0 # log1p(0)
    np.testing.assert_almost_equal(transformed['Log_Amount'].iloc[1], np.log(100), decimal=5)