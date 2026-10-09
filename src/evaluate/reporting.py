import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_recall_curve,
)

from src.evaluate.metrics import compare_thresholds, compute_metrics


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 120,
) -> dict:
    """Plot and save a precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auc = float(np.trapz(precision, recall)) if len(precision) > 1 else 0.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(
        ax=ax, name=f"PR-AUC = {auc:.4f}"
    )
    ax.set_title("Precision-Recall Curve (Test Set)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "pr_auc": auc,
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_path: str,
    figsize: tuple = (6, 6),
    dpi: int = 120,
) -> dict:
    """Plot and save a confusion matrix at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.4f})")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {"confusion_matrix": cm.tolist()}


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float,
    fn_multiplier: float,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 120,
) -> dict:
    """Plot precision/recall/F1 and business cost across thresholds."""
    rows = compare_thresholds(y_true, y_prob, amounts, fp_cost, fn_multiplier)

    thresholds = np.array([r["Threshold"] for r in rows])
    precision = np.array([r["Precision"] for r in rows])
    recall = np.array([r["Recall"] for r in rows])
    f1 = np.array([r["F1"] for r in rows])
    cost = np.array([r["Total_Cost"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

    ax1.plot(thresholds, precision, label="Precision", linewidth=1.5)
    ax1.plot(thresholds, recall, label="Recall", linewidth=1.5)
    ax1.plot(thresholds, f1, label="F1", linewidth=1.5)
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 vs Threshold (Validation Set)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        thresholds, cost, label="Total Business Cost", color="tab:red", linewidth=1.5
    )
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Total Cost ($)")
    ax2.set_title("Business Cost vs Threshold (Validation Set)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "thresholds": thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "total_cost": cost.tolist(),
    }


def plot_feature_importance(
    pipeline,
    sample_X: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
    figsize: tuple = (10, 6),
    dpi: int = 120,
) -> dict:
    """Plot XGBoost built-in feature importance (gain).

    Extracts human-readable feature names from the pipeline's
    preprocessing steps so the plot is interpretable.
    """
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_names = _extract_feature_names(pipeline, sample_X)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in sorted_idx]
    values = [float(importances[i]) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.barh(range(len(names)), values, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain Importance")
    ax.set_title("XGBoost Feature Importance (Top {} by Gain)".format(top_n))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "top_features": [{"feature": n, "importance": v} for n, v in zip(names, values)]
    }


def _extract_feature_names(pipeline, sample_X: pd.DataFrame) -> list[str]:
    """Transform a small sample through preprocessing steps to recover output names."""
    from sklearn.pipeline import Pipeline as SkPipeline

    if not isinstance(pipeline, SkPipeline):
        return []

    X_temp = sample_X.copy()
    for name, step in pipeline.steps[:-1]:
        X_temp = step.transform(X_temp)

    if hasattr(X_temp, "columns"):
        return list(X_temp.columns)
    return [f"feature_{i}" for i in range(X_temp.shape[1])]


def generate_evaluation_plots(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    val_y: np.ndarray,
    val_prob: np.ndarray,
    val_amounts: np.ndarray,
    test_amounts: np.ndarray,
    threshold: float,
    fp_cost: float,
    fn_multiplier: float,
    pipeline,
    sample_X: pd.DataFrame,
    output_dir: str,
    dpi: int = 120,
) -> dict:
    """Generate all evaluation plots and return a summary dict."""
    os.makedirs(output_dir, exist_ok=True)

    pr_result = plot_pr_curve(
        y_test, y_prob, os.path.join(output_dir, "pr_curve.png"), dpi=dpi
    )
    cm_result = plot_confusion_matrix(
        y_test,
        y_prob,
        threshold,
        os.path.join(output_dir, "confusion_matrix.png"),
        dpi=dpi,
    )
    sens_result = plot_threshold_sensitivity(
        val_y,
        val_prob,
        val_amounts,
        fp_cost,
        fn_multiplier,
        os.path.join(output_dir, "threshold_sensitivity.png"),
        dpi=dpi,
    )
    fi_result = plot_feature_importance(
        pipeline, sample_X, os.path.join(output_dir, "feature_importance.png"), dpi=dpi
    )

    return {
        "pr_curve": pr_result,
        "confusion_matrix": cm_result,
        "threshold_sensitivity": sens_result,
        "feature_importance": fi_result,
    }


def generate_evaluation_report(
    train_df,
    val_df,
    test_df,
    y_val,
    val_probs,
    val_amounts,
    y_test,
    y_prob,
    test_amounts,
    threshold,
    fp_cost,
    fn_multiplier,
    comparison,
    threshold_rows,
    output_path: str,
) -> dict:
    """Generate a markdown evaluation report separating validation from test.

    The report clearly distinguishes VALIDATION results (used for threshold
    optimisation) from FINAL TEST results (held out until the end).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_n, val_n, test_n = len(train_df), len(val_df), len(test_df)
    neg = int((train_df["Class"] == 0).sum())
    pos = int((train_df["Class"] == 1).sum())
    best_row = min(threshold_rows, key=lambda r: r["Total_Cost"])
    val_metrics = compute_metrics(
        y_val,
        val_probs,
        threshold=best_row["Threshold"],
        amounts=val_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
    )
    test_metrics = compute_metrics(
        y_test,
        y_prob,
        threshold=threshold,
        amounts=test_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
    )

    L = ["# Fraud Detection Evaluation Report", ""]
    L.append("## 1. Dataset")
    L.append("")
    L.append("| Split | Rows | Fraud | Fraud rate |")
    L.append("|---|---|---|---|")
    L.append(f"| Train | {train_n:,} | {pos:,} | {pos / train_n * 100:.3f}% |")
    val_fraud = int(val_df["Class"].sum())
    test_fraud = int(test_df["Class"].sum())
    val_rate = val_df["Class"].mean() * 100
    test_rate = test_df["Class"].mean() * 100
    L.append(f"| Validation | {val_n:,} | {val_fraud:,} | {val_rate:.3f}% |")
    L.append(f"| Test | {test_n:,} | {test_fraud:,} | {test_rate:.3f}% |")
    L.append("")
    L.append("## 2. Class imbalance")
    L.append("")
    L.append(
        f"- scale_pos_weight (train neg/pos): {neg:,} / {pos:,} = {neg / pos:.1f}x"
    )
    L.append("")
    L.append("## 3. Validation set (threshold optimisation only)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Selected threshold | {best_row['Threshold']:.4f} |")
    L.append(f"| Precision | {val_metrics['Precision']:.3f} |")
    L.append(f"| Recall | {val_metrics['Recall']:.3f} |")
    L.append(f"| F1 | {val_metrics['F1']:.3f} |")
    L.append(f"| Total cost | ${best_row['Total_Cost']:.2f} |")
    L.append("")
    L.append("## 4. Final TEST set (used exactly once, after all modelling decisions)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| ROC-AUC | {test_metrics['ROC_AUC']:.4f} |")
    L.append(f"| PR-AUC | {test_metrics['PR_AUC']:.4f} |")
    L.append(f"| Precision | {test_metrics['Precision']:.4f} |")
    L.append(f"| Recall | {test_metrics['Recall']:.4f} |")
    L.append(f"| F1 | {test_metrics['F1']:.4f} |")
    L.append(f"| Decision threshold | {threshold:.4f} |")
    L.append(f"| False Positives | {test_metrics['False_Positives']} |")
    L.append(f"| False Negatives | {test_metrics['False_Negatives']} |")
    L.append(f"| Total cost | ${test_metrics['Total_Cost']:.2f} |")
    L.append("")
    L.append("## 5. Model comparison (test set, PR-AUC)")
    L.append("")
    L.append("| Model | PR-AUC | Precision | Recall | F1 | Threshold | Total cost |")
    L.append("|---|---|---|---|---|---|---|")
    for name, info in comparison.items():
        m = info["metrics"]
        L.append(
            f"| {name} | {m['PR_AUC']:.4f} | {m['Precision']:.4f} | "
            f"{m['Recall']:.4f} | {m['F1']:.4f} | {info['threshold']:.4f} | "
            f"${m['Total_Cost']:.2f} |"
        )
    L.append("")
    L.append("## 6. Interpretation")
    L.append("")
    L.append(
        "- PR-AUC is the primary metric because fraud is "
        f"~{neg / pos:.0f}x rarer than legitimate."
    )
    L.append(
        "- The decision threshold was optimised on the VALIDATION set only, then frozen"
    )
    L.append("  and applied to the held-out TEST set exactly once.")
    L.append(
        "- Business costs assumed for this study are hypothetical "
        "($10 per false positive)."
    )
    L.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(L))

    return {
        "report_path": output_path,
        "validation": {
            "selected_threshold": best_row["Threshold"],
            "total_cost": best_row["Total_Cost"],
        },
        "test": test_metrics,
        "comparison": comparison,
    }
