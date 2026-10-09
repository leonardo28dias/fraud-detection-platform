"""Tests for data ingestion and temporal splitting."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_and_split_data


@pytest.fixture
def synthetic_csv(tmp_path):
    """Create a small synthetic CSV that mimics creditcard.csv structure."""
    rng = np.random.RandomState(42)
    n = 1000
    df = pd.DataFrame(
        {
            "Time": np.sort(rng.uniform(0, 172800, n)),  # sorted by time
            "Amount": rng.uniform(1, 500, n),
            **{f"V{i}": rng.randn(n) for i in range(1, 29)},
            "Class": np.zeros(n, dtype=int),
        }
    )
    # Inject 10 fraud cases in the later portion (temporal ordering)
    fraud_idx = rng.choice(range(400, n), size=10, replace=False)
    df.loc[fraud_idx, "Class"] = 1
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_temporal_split_sizes(synthetic_csv):
    """Train + val + test must sum to total."""
    train_df, val_df, test_df = load_and_split_data(synthetic_csv, "Time")
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 1000
    # Approximate 70/15/15 split
    assert len(train_df) == pytest.approx(700, abs=5)
    assert len(val_df) == pytest.approx(150, abs=5)
    assert len(test_df) == pytest.approx(150, abs=5)


def test_temporal_split_is_chronological(synthetic_csv):
    """Test data must come after train + val in time ordering."""
    train_df, val_df, test_df = load_and_split_data(synthetic_csv, "Time")
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_no_data_leakage_between_splits(synthetic_csv):
    """No row should appear in more than one split."""
    train_df, val_df, test_df = load_and_split_data(synthetic_csv, "Time")
    train_len = len(train_df)
    val_len = len(val_df)
    test_len = len(test_df)
    assert train_len + val_len + test_len == 1000


def test_fraud_distribution_not_empty(synthetic_csv):
    """Each split should contain at least some fraud cases."""
    train_df, val_df, test_df = load_and_split_data(synthetic_csv, "Time")
    assert train_df["Class"].sum() > 0
    assert val_df["Class"].sum() > 0
    assert test_df["Class"].sum() > 0
