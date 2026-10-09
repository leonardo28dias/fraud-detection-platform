"""Tests for feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import LogAmountTransformer, TemporalFeatureExtractor


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Time": [
                0,
                3600,
                43200,
                86400,
                129600,
            ],  # midnight, 1am, noon, midnight+1d, 36h
            "Amount": [0.0, 50.0, 1000.0, 0.01, 50000.0],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "V2": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    )


def test_temporal_feature_extractor_drops_time(sample_df):
    extractor = TemporalFeatureExtractor()
    transformed = extractor.fit_transform(sample_df)
    assert "Time" not in transformed.columns


def test_temporal_feature_extractor_adds_cyclical_features(sample_df):
    extractor = TemporalFeatureExtractor()
    transformed = extractor.fit_transform(sample_df)
    assert "Hour_Sin" in transformed.columns
    assert "Hour_Cos" in transformed.columns


def test_temporal_feature_extractor_sin_cos_range(sample_df):
    """Sin and cos features must be in [-1, 1]."""
    extractor = TemporalFeatureExtractor()
    transformed = extractor.fit_transform(sample_df)
    assert transformed["Hour_Sin"].min() >= -1.0
    assert transformed["Hour_Sin"].max() <= 1.0
    assert transformed["Hour_Cos"].min() >= -1.0
    assert transformed["Hour_Cos"].max() <= 1.0


def test_log_amount_transformer_drops_amount(sample_df):
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(sample_df)
    assert "Amount" not in transformed.columns


def test_log_amount_transformer_adds_log_amount(sample_df):
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(sample_df)
    assert "Log_Amount" in transformed.columns


def test_log_amount_transformer_values(sample_df):
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(sample_df)
    expected = np.log1p(sample_df["Amount"].values)
    np.testing.assert_almost_equal(transformed["Log_Amount"].values, expected)


def test_log_amount_handles_zero():
    """log1p(0) = 0, important for zero-amount transactions."""
    df = pd.DataFrame({"Amount": [0.0, 1.0]})
    transformer = LogAmountTransformer()
    transformed = transformer.fit_transform(df)
    assert transformed["Log_Amount"].iloc[0] == 0.0


def test_transformer_preserves_other_columns(sample_df):
    """V1, V2 columns should survive both transformations."""
    extractor = TemporalFeatureExtractor()
    transformer = LogAmountTransformer()
    result = transformer.fit_transform(extractor.fit_transform(sample_df))
    assert "V1" in result.columns
    assert "V2" in result.columns
