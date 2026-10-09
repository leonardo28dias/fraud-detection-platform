import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def generate_shap_summary(
    pipeline: Pipeline,
    X: pd.DataFrame,
    output_dir: str,
    max_display: int = 20,
) -> None:
    """Generate a global SHAP feature-importance summary plot.

    The pipeline's preprocessing steps are applied manually to produce
    the transformed feature matrix that the classifier sees.
    Feature names are recovered from the last DataFrame-producing step
    (the custom transformers) and passed to SHAP so the summary shows
    meaningful names rather than f0/f1/etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Walk the preprocessing steps to recover output feature names
    X_transformed = X.copy()
    feature_names: list[str] = list(X.columns)
    for name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)
        if hasattr(X_transformed, "columns"):
            feature_names = list(X_transformed.columns)

    model = pipeline.named_steps["classifier"]

    # TreeExplainer is the correct explainer for XGBoost models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    # In shap >= 0.42 the callable API returns an Explanation object
    if hasattr(shap_values, "values"):
        explanation = shap_values
    else:
        explanation = (
            shap.Explanation(
                values=shap_values,
                data=X_transformed,
                feature_names=feature_names,
            )
            if not isinstance(shap_values, shap.Explanation)
            else shap_values
        )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        explanation,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance (Test Set Sample)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "shap_summary.png"), bbox_inches="tight", dpi=120
    )
    plt.close()
    logger.info("SHAP summary plot saved to %s", output_dir)
