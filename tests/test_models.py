"""Tests for the model trainer.

Covers scale_pos_weight, pipeline building, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.trainer import (
    build_pipeline,
    compute_scale_pos_weight,
    tune_hyperparameters,
)


@pytest.fixture
def balanced_train_data():
    rng = np.random.RandomState(42)
    n = 1000
    X = rng.randn(n, 5)
    y = np.zeros(n, dtype=int)
    y[:50] = 1
    return X, y


@pytest.fixture
def imbalanced_train_data():
    rng = np.random.RandomState(42)
    n = 1000
    X = rng.randn(n, 5)
    y = np.zeros(n, dtype=int)
    y[:5] = 1
    return X, y


@pytest.fixture
def tabular_train_data():
    """DataFrame with the creditcard.csv column structure for full-pipeline tests."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "Time": np.sort(rng.uniform(0, 172800, n)),
            "Amount": rng.uniform(1, 500, n),
            **{f"V{i}": rng.randn(n) for i in range(1, 29)},
            "Class": np.zeros(n, dtype=int),
        }
    )
    df.loc[:10, "Class"] = 1
    return df.drop(columns=["Class"]), df["Class"].values


# ── compute_scale_pos_weight ──────────────────────────────────────────────


def test_scale_pos_weight_balanced():
    y = np.array([0, 0, 1, 1])
    ratio = compute_scale_pos_weight(y)
    assert ratio == 1.0


def test_scale_pos_weight_imbalanced(balanced_train_data):
    _, y = balanced_train_data
    ratio = compute_scale_pos_weight(y)
    expected = 950 / 50  # 950 negatives, 50 positives
    assert ratio == pytest.approx(expected, rel=1e-4)


def test_scale_pos_weight_uses_only_train_data(imbalanced_train_data):
    """The ratio must be computed from the training labels only."""
    X, y = imbalanced_train_data
    ratio = compute_scale_pos_weight(y)
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    assert ratio == pytest.approx(n_neg / n_pos)


def test_scale_pos_weight_no_positives():
    """If there are no positive samples, weight should be 1.0 (safe default)."""
    y = np.zeros(100, dtype=int)
    ratio = compute_scale_pos_weight(y)
    assert ratio == 1.0


# ── build_pipeline ────────────────────────────────────────────────────────


def test_build_pipeline_xgboost():
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "scale_pos_weight": 100,
        "random_state": 42,
        "n_jobs": -1,
    }
    pipeline = build_pipeline("xgboost", params)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["classifier"], XGBClassifier)


def test_build_pipeline_logistic_regression():
    params = {"max_iter": 200, "random_state": 42, "class_weight": "balanced"}
    pipeline = build_pipeline("logistic_regression", params)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_build_pipeline_invalid_type():
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_pipeline("random_forest", {})


def test_pipeline_has_scaler():
    """The StandardScaler step prevents leakage by fitting on train only."""
    pipeline = build_pipeline("xgboost", {"random_state": 42})
    step_names = [name for name, _ in pipeline.steps]
    assert "scaler" in step_names


def test_pipeline_has_feature_transforms():
    pipeline = build_pipeline("xgboost", {"random_state": 42})
    step_names = [name for name, _ in pipeline.steps]
    assert "temporal_features" in step_names
    assert "amount_transform" in step_names


# ── tune_hyperparameters ───────────────────────────────────────────────────


def test_tune_hyperparameters_returns_pipeline(balanced_train_data):
    """GridSearchCV should return a fitted Pipeline and best params."""
    X, y = balanced_train_data
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    result, best_params = tune_hyperparameters(
        pipeline,
        X,
        y,
        param_grid={"C": [0.1, 1.0]},
        cv_folds=3,
        scoring="average_precision",
    )
    assert isinstance(result, Pipeline)
    assert best_params is not None
    assert "C" in best_params


def test_tune_hyperparameters_logs_best_params(balanced_train_data, caplog):
    """The best parameters should be logged."""
    X, y = balanced_train_data
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    caplog.set_level("INFO")
    tune_hyperparameters(
        pipeline,
        X,
        y,
        param_grid={"C": [0.1, 1.0]},
        cv_folds=3,
    )
    assert "Best params" in caplog.text


def test_build_pipeline_xgboost_trains(tabular_train_data):
    """Full XGBoost pipeline should train on tabular data with correct columns."""
    X, y = tabular_train_data
    pipeline = build_pipeline(
        "xgboost",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 18,
            "random_state": 42,
            "n_jobs": -1,
        },
    )
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    assert probs.shape == (200, 2)
