import logging

import numpy as np

logger = logging.getLogger(__name__)


def optimize_business_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float,
    fn_multiplier: float,
) -> float:
    """Find the probability threshold that minimises total expected business cost.

    Cost = (False Positives * fp_cost) + (False Negatives * amounts * fn_multiplier)

    The search is a simple grid over 100 thresholds between 0.01 and 0.99.
    Thresholds should be selected on the **validation** set, never on the
    test set.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_cost = float("inf")
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        fps = int(np.sum((y_pred == 1) & (y_true == 0)))
        missed_amounts = amounts[(y_pred == 0) & (y_true == 1)]
        fn_cost = float(np.sum(missed_amounts) * fn_multiplier)
        total_cost = (fps * fp_cost) + fn_cost

        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = float(thresh)

    logger.info(
        "Best threshold: %.4f (validation cost = $%.2f, fp_cost=$%.1f, fn_mult=%.1f)",
        best_thresh,
        best_cost,
        fp_cost,
        fn_multiplier,
    )
    return best_thresh
