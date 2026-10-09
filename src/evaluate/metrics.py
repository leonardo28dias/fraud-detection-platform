from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    amounts: np.ndarray | None = None,
    fp_cost: float = 10.0,
    fn_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Compute classification and business-cost metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    result: Dict[str, Any] = {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "False_Positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "False_Negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "True_Positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        "True_Negatives": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "Threshold": float(threshold),
    }

    if amounts is not None:
        fn_amounts = amounts[(y_pred == 0) & (y_true == 1)]
        fn_cost = float(np.sum(fn_amounts) * fn_multiplier)
        fp_cost_total = result["False_Positives"] * fp_cost
        total_cost = fp_cost_total + fn_cost
        result["False_Positive_Cost"] = float(fp_cost_total)
        result["False_Negative_Cost"] = float(fn_cost)
        result["Total_Cost"] = float(total_cost)

    return result


def compare_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float,
    fn_multiplier: float,
    thresholds: np.ndarray | None = None,
) -> list[Dict[str, Any]]:
    """Evaluate metrics across a range of thresholds for sensitivity analysis.

    Returns a list of metric dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        row = compute_metrics(
            y_true,
            y_prob,
            threshold=t,
            amounts=amounts,
            fp_cost=fp_cost,
            fn_multiplier=fn_multiplier,
        )
        rows.append(row)
    return rows
