import logging
from typing import Any, Dict

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Calculate the scale_pos_weight for imbalanced classification.

    Uses ONLY training labels so no information leaks from
    validation or test sets.

    scale_pos_weight = (count of negative samples) / (count of positive samples)
    """
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    if n_pos == 0:
        logger.warning(
            "No positive samples in training data; scale_pos_weight set to 1.0"
        )
        return 1.0
    ratio = n_neg / n_pos
    logger.info(
        f"Class imbalance — negative={n_neg}, positive={n_pos}, "
        f"scale_pos_weight={ratio:.2f}"
    )
    return ratio


def build_pipeline(model_type: str, params: Dict[str, Any]) -> Pipeline:
    """Construct the modeling pipeline with feature engineering and classifier.

    The sklearn Pipeline ensures all preprocessing is fitted only on
    training data, preventing data leakage.
    """
    from sklearn.preprocessing import StandardScaler

    from src.features.engineering import LogAmountTransformer, TemporalFeatureExtractor

    logger.info(f"Building pipeline with model: {model_type}")

    if model_type == "xgboost":
        classifier = xgb.XGBClassifier(**params)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("temporal_features", TemporalFeatureExtractor(time_col="Time")),
            ("amount_transform", LogAmountTransformer(amount_col="Amount")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    return pipeline


def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: Any,
    y_train: np.ndarray,
    param_grid: Dict[str, list],
    cv_folds: int = 3,
    scoring: str = "average_precision",
) -> Pipeline:
    """Tune hyperparameters using time-series-aware cross-validation.

    Uses TimeSeriesSplit so that each fold only trains on data
    chronologically *before* the fold it validates on — no future
    leakage during hyperparameter selection.

    The test set is never touched during tuning.
    """
    logger.info(
        f"Starting hyperparameter tuning ({len(param_grid)} params, {cv_folds} folds)"
    )

    # Prefix grid keys with 'classifier__' to target the pipeline step
    prefixed_grid = {f"classifier__{k}": v for k, v in param_grid.items()}

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=prefixed_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=1,  # XGBoost already uses n_jobs=-1; avoid CPU oversubscription
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")

    # Strip the 'classifier__' prefix for readability
    best_params = {
        k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()
    }
    return grid_search.best_estimator_, best_params
