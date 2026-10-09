"""Tests for threshold optimization."""

import numpy as np
import pytest

from src.models.thresholding import optimize_business_threshold


@pytest.fixture
def sample_predictions():
    rng = np.random.RandomState(42)
    y_true = np.array([0] * 900 + [1] * 100)
    y_prob = np.concatenate(
        [
            rng.uniform(0.0, 0.6, 900),
            rng.uniform(0.3, 1.0, 100),
        ]
    )
    amounts = np.concatenate(
        [
            rng.uniform(10, 200, 900),
            rng.uniform(100, 2000, 100),
        ]
    )
    rng.shuffle(y_prob)  # shuffle to simulate real-world ordering
    return y_true, y_prob, amounts


def test_threshold_in_valid_range(sample_predictions):
    y_true, y_prob, amounts = sample_predictions
    thresh = optimize_business_threshold(y_true, y_prob, amounts, 10.0, 1.0)
    assert 0.01 <= thresh <= 0.99


def test_threshold_minimizes_cost(sample_predictions):
    """The returned threshold should have lower or equal cost than 0.5."""
    y_true, y_prob, amounts = sample_predictions
    best = optimize_business_threshold(y_true, y_prob, amounts, 10.0, 1.0)

    def cost_at(t):
        y_pred = (y_prob >= t).astype(int)
        fps = np.sum((y_pred == 1) & (y_true == 0))
        missed = amounts[(y_pred == 0) & (y_true == 1)]
        return fps * 10.0 + np.sum(missed)

    baseline_cost = cost_at(0.5)
    optimized_cost = cost_at(best)
    assert optimized_cost <= baseline_cost


def test_threshold_handles_no_fraud():
    """If there are no fraud cases, any threshold should be returned safely."""
    y_true = np.zeros(100, dtype=int)
    y_prob = np.random.rand(100)
    amounts = np.random.uniform(10, 100, 100)
    thresh = optimize_business_threshold(y_true, y_prob, amounts, 10.0, 1.0)
    assert 0.01 <= thresh <= 0.99


def test_threshold_returns_float(sample_predictions):
    y_true, y_prob, amounts = sample_predictions
    thresh = optimize_business_threshold(y_true, y_prob, amounts, 10.0, 1.0)
    assert isinstance(thresh, float)
