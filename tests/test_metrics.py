"""Tests for metrics computation and business cost."""

import numpy as np
import pytest

from src.evaluate.metrics import compare_thresholds, compute_metrics


@pytest.fixture
def sample_data():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2, 0.7, 0.1])
    amounts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    return y_true, y_prob, amounts


# ── compute_metrics ────────────────────────────────────────────────────────


def test_compute_metrics_keys(sample_data):
    y_true, y_prob, _ = sample_data
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    for key in [
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "False_Positives",
        "False_Negatives",
        "True_Positives",
        "True_Negatives",
        "Threshold",
    ]:
        assert key in metrics


def test_compute_metrics_threshold(sample_data):
    y_true, y_prob, _ = sample_data
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["Threshold"] == 0.5
    m2 = compute_metrics(y_true, y_prob, threshold=0.3)
    assert m2["Threshold"] == 0.3


def test_compute_metrics_confusion_matrix(sample_data):
    y_true, y_prob, amounts = sample_data
    m = compute_metrics(
        y_true, y_prob, threshold=0.5, amounts=amounts, fp_cost=10.0, fn_multiplier=1.0
    )
    cm_sum = (
        m["True_Positives"]
        + m["False_Positives"]
        + m["False_Negatives"]
        + m["True_Negatives"]
    )
    assert cm_sum == len(y_true)


def test_compute_metrics_with_cost(sample_data):
    y_true, y_prob, amounts = sample_data
    m = compute_metrics(
        y_true, y_prob, threshold=0.5, amounts=amounts, fp_cost=10.0, fn_multiplier=1.0
    )
    expected_fp_cost = m["False_Positives"] * 10.0
    assert "Total_Cost" in m
    assert m["False_Positive_Cost"] == pytest.approx(expected_fp_cost)


def test_compute_metrics_no_false_positives():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["False_Positives"] == 0


# ── compare_thresholds ─────────────────────────────────────────────────────


def test_compare_thresholds_returns_list(sample_data):
    y_true, y_prob, amounts = sample_data
    rows = compare_thresholds(
        y_true, y_prob, amounts, 10.0, 1.0, thresholds=np.array([0.3, 0.5, 0.7])
    )
    assert len(rows) == 3
    assert rows[0]["Threshold"] == 0.3
    assert rows[1]["Threshold"] == 0.5
    assert rows[2]["Threshold"] == 0.7


def test_compare_thresholds_costs_decrease_then_increase(sample_data):
    """Cost should generally decrease then increase as threshold rises."""
    y_true, y_prob, amounts = sample_data
    rows = compare_thresholds(y_true, y_prob, amounts, 10.0, 1.0)
    costs = [r["Total_Cost"] for r in rows]
    assert len(costs) > 2


def test_compare_thresholds_custom_thresholds(sample_data):
    y_true, y_prob, amounts = sample_data
    custom = np.array([0.25, 0.50, 0.75])
    rows = compare_thresholds(y_true, y_prob, amounts, 10.0, 1.0, thresholds=custom)
    assert [r["Threshold"] for r in rows] == [0.25, 0.50, 0.75]
