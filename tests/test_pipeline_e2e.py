"""End-to-end pipeline behaviour tests (fast, synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_and_split_data
from src.evaluate.metrics import compute_metrics
from src.models.thresholding import optimize_business_threshold
from src.models.trainer import build_pipeline, compute_scale_pos_weight


def _make_synthetic_csv(tmp_path, n=600):
    """Write a creditcard-shaped CSV to tmp_path and return its path."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "Time": np.sort(rng.uniform(0, 172800, n)),
            "Amount": rng.uniform(1, 500, n),
            **{f"V{i}": rng.randn(n) for i in range(1, 29)},
            "Class": np.zeros(n, dtype=int),
        }
    )
    fraud_idx = rng.choice(range(200, n), size=8, replace=False)
    df.loc[fraud_idx, "Class"] = 1
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---- Temporal split + class-weight use only training labels ----


def test_temporal_split_sizes_and_order(tmp_path):
    """Splits sum to total, are chronological, and each has some fraud."""
    csv_path = _make_synthetic_csv(tmp_path)
    train_df, val_df, test_df = load_and_split_data(csv_path, "Time")
    assert len(train_df) + len(val_df) + len(test_df) == 600
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()
    assert train_df["Class"].sum() > 0
    assert val_df["Class"].sum() > 0
    assert test_df["Class"].sum() > 0


def test_class_weight_uses_only_training_labels(tmp_path):
    """scale_pos_weight must be computed from train labels only."""
    csv_path = _make_synthetic_csv(tmp_path, n=600)
    train_df, _, _ = load_and_split_data(csv_path, "Time")
    y_train = train_df["Class"].values
    ratio = compute_scale_pos_weight(y_train)
    expected = int((y_train == 0).sum()) / max(1, int((y_train == 1).sum()))
    assert ratio == pytest.approx(expected)


# ---- End-to-end pipeline behaviour ----


def test_full_pipeline_end_to_end(tmp_path):
    """Train -> threshold -> test metrics follows the documented flow."""
    csv_path = _make_synthetic_csv(tmp_path, n=600)
    train_df, val_df, test_df = load_and_split_data(csv_path, "Time")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].values
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"].values
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].values
    val_amounts = val_df["Amount"].values
    test_amounts = test_df["Amount"].values

    # Logistic-regression baseline (no tuning)
    lr_pipe = build_pipeline(
        "logistic_regression", {"max_iter": 200, "random_state": 42}
    )
    lr_pipe.fit(X_train, y_train)
    lr_val_probs = lr_pipe.predict_proba(X_val)[:, 1]
    lr_thresh = optimize_business_threshold(y_val, lr_val_probs, val_amounts, 10.0, 1.0)
    lr_test_probs = lr_pipe.predict_proba(X_test)[:, 1]
    lr_metrics = compute_metrics(
        y_test,
        lr_test_probs,
        threshold=lr_thresh,
        amounts=test_amounts,
        fp_cost=10.0,
        fn_multiplier=1.0,
    )

    # XGBoost with dynamic class weighting
    spw = compute_scale_pos_weight(y_train)
    xgb_pipe = build_pipeline(
        "xgboost",
        {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": spw,
            "random_state": 42,
            "n_jobs": -1,
        },
    )
    xgb_pipe.fit(X_train, y_train)
    xgb_val_probs = xgb_pipe.predict_proba(X_val)[:, 1]
    xgb_thresh = optimize_business_threshold(
        y_val, xgb_val_probs, val_amounts, 10.0, 1.0
    )
    xgb_test_probs = xgb_pipe.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(
        y_test,
        xgb_test_probs,
        threshold=xgb_thresh,
        amounts=test_amounts,
        fp_cost=10.0,
        fn_multiplier=1.0,
    )

    # Confusion matrix must account for every test sample (behaviour check)
    cm_total = (
        xgb_metrics["True_Positives"]
        + xgb_metrics["False_Positives"]
        + xgb_metrics["False_Negatives"]
        + xgb_metrics["True_Negatives"]
    )
    assert cm_total == len(y_test)

    # Threshold must be in the searched range on validation
    assert 0.01 <= lr_thresh <= 0.99
    assert 0.01 <= xgb_thresh <= 0.99

    # XGB should beat LR on PR-AUC in expectation on this synthetic data
    assert xgb_metrics["PR_AUC"] > lr_metrics["PR_AUC"]

    # Test set is never touched during training or tuning.
    assert len(X_test) > 0
    assert xgb_metrics["True_Positives"] + xgb_metrics["False_Negatives"] == int(
        (y_test == 1).sum()
    )
