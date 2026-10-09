"""End-to-end fraud detection pipeline.

Execution flow:
  1. Load data and split temporally (train / val / test)
  2. Compute scale_pos_weight from training labels only
  3. Train Logistic Regression baseline (same preprocessing)
  4. Train XGBoost with class weighting (+ optional hyperparameter tuning)
  5. Optimize decision threshold on the validation set (cost-sensitive)
  6. Evaluate both models on the held-out test set
  7. Generate evaluation plots (PR curve, confusion matrix, threshold sensitivity)
  8. Generate SHAP global explainability summary
  9. Save model artifacts, metrics, and comparison report
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.ingestion import load_and_split_data
from src.evaluate.explainability import generate_shap_summary
from src.evaluate.metrics import compare_thresholds, compute_metrics
from src.evaluate.reporting import generate_evaluation_plots, generate_evaluation_report
from src.models.thresholding import optimize_business_threshold
from src.models.trainer import (
    build_pipeline,
    compute_scale_pos_weight,
    tune_hyperparameters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_evaluate_model(
    model_type: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_amounts: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_amounts: np.ndarray,
    fp_cost: float,
    fn_multiplier: float,
    tune_config: dict | None = None,
) -> dict:
    """Train a model, optimize its threshold, and evaluate on the test set.

    Returns a dict with the trained pipeline, metrics, threshold, and
    validation predictions (for plots).
    """
    logger.info(f"--- Training {model_type} ---")

    # Optional hyperparameter tuning (TimeSeriesSplit on training data only)
    best_params = None
    if tune_config and tune_config.get("enabled"):
        base_pipeline = build_pipeline(model_type, params)
        pipeline, best_params = tune_hyperparameters(
            base_pipeline,
            X_train,
            y_train,
            param_grid=tune_config["param_grid"],
            cv_folds=tune_config["cv_folds"],
            scoring=tune_config["scoring"],
        )
        logger.info(f"Tuned {model_type} pipeline ready")
    else:
        pipeline = build_pipeline(model_type, params)
        pipeline.fit(X_train, y_train)

    # Threshold optimization on VALIDATION set only
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    best_thresh = optimize_business_threshold(
        y_true=y_val,
        y_prob=val_probs,
        amounts=val_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
    )
    logger.info(f"{model_type} optimal threshold (val): {best_thresh:.4f}")

    # Final evaluation on held-out TEST set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(
        y_test,
        test_probs,
        threshold=best_thresh,
        amounts=test_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
    )
    logger.info(
        f"{model_type} test metrics: "
        f"ROC-AUC={metrics['ROC_AUC']:.4f}, "
        f"PR-AUC={metrics['PR_AUC']:.4f}, "
        f"Precision={metrics['Precision']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"F1={metrics['F1']:.4f}"
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "threshold": best_thresh,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "best_params": best_params,
    }


def error_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    time_values: np.ndarray,
    threshold: float,
    output_path: str,
) -> dict:
    """Analyse false positives and false negatives."""
    y_pred = (y_prob >= threshold).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    analysis = {
        "false_positives": {
            "count": int(np.sum(fp_mask)),
            "avg_amount": float(np.mean(amounts[fp_mask])) if np.any(fp_mask) else 0.0,
            "median_amount": float(np.median(amounts[fp_mask]))
            if np.any(fp_mask)
            else 0.0,
            "avg_time": float(np.mean(time_values[fp_mask]))
            if np.any(fp_mask)
            else 0.0,
        },
        "false_negatives": {
            "count": int(np.sum(fn_mask)),
            "avg_amount": float(np.mean(amounts[fn_mask])) if np.any(fn_mask) else 0.0,
            "median_amount": float(np.median(amounts[fn_mask]))
            if np.any(fn_mask)
            else 0.0,
            "avg_time": float(np.mean(time_values[fn_mask]))
            if np.any(fn_mask)
            else 0.0,
        },
        "true_positives": {
            "count": int(np.sum((y_pred == 1) & (y_true == 1))),
            "avg_amount": float(np.mean(amounts[(y_pred == 1) & (y_true == 1)]))
            if np.any((y_pred == 1) & (y_true == 1))
            else 0.0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Error analysis saved to {output_path}")
    logger.info(
        f"False Positives: {analysis['false_positives']['count']} "
        f"(avg amount: ${analysis['false_positives']['avg_amount']:.2f}), "
        f"False Negatives: {analysis['false_negatives']['count']} "
        f"(avg amount: ${analysis['false_negatives']['avg_amount']:.2f})"
    )

    return analysis


def main():
    config = load_config()

    artifact_dir = config["model"]["artifact_dir"]
    reports_dir = "reports/"
    results_dir = "results/"
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fp_cost = config["business_cost"]["false_positive_cost"]
    fn_multiplier = config["business_cost"]["false_negative_multiplier"]

    # ── 1. Ingestion & Temporal Split ──────────────────────────────────
    train_df, val_df, test_df = load_and_split_data(
        filepath=config["data"]["raw_path"],
        time_col=config["data"]["time_col"],
        val_size=config["data"]["val_size"],
        test_size=config["data"]["test_size"],
    )

    target_col = config["data"]["target_col"]
    time_col = config["data"]["time_col"]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    val_amounts = val_df["Amount"].values

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    test_amounts = test_df["Amount"].values
    test_times = test_df[time_col].values

    # ── 2. Class imbalance ratio (TRAINING DATA ONLY) ──────────────────
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # ── 3. Baseline: Logistic Regression ───────────────────────────────
    lr_params = dict(config["model"]["baseline"]["params"])
    lr_result = train_evaluate_model(
        model_type="logistic_regression",
        params=lr_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_amounts=val_amounts,
        X_test=X_test,
        y_test=y_test,
        test_amounts=test_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
    )

    # ── 4. XGBoost with class weighting (+ tuning) ───────────────────────
    xgb_params = dict(config["model"]["params"])
    xgb_params["scale_pos_weight"] = scale_pos_weight

    tune_config = config["model"].get("tune", {})
    xgb_result = train_evaluate_model(
        model_type="xgboost",
        params=xgb_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_amounts=val_amounts,
        X_test=X_test,
        y_test=y_test,
        test_amounts=test_amounts,
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
        tune_config=tune_config,
    )

    # ── 5. Comparison ──────────────────────────────────────────────────
    comparison = {
        "logistic_regression": {
            "threshold": lr_result["threshold"],
            "metrics": lr_result["metrics"],
        },
        "xgboost": {
            "threshold": xgb_result["threshold"],
            "metrics": xgb_result["metrics"],
        },
    }
    logger.info(
        f"Comparison (test set, PR-AUC): "
        f"LR={lr_result['metrics']['PR_AUC']:.4f}  "
        f"XGBoost={xgb_result['metrics']['PR_AUC']:.4f}"
    )

    shap_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    logger.info("Generating SHAP explanations...")
    generate_shap_summary(xgb_result["pipeline"], shap_sample, reports_dir)

    # ── 7. Evaluation plots (validation for sensitivity, test for PR/CM) ─
    generate_evaluation_plots(
        y_test=y_test,
        y_prob=xgb_result["test_probs"],
        val_y=y_val,
        val_prob=xgb_result["val_probs"],
        val_amounts=val_amounts,
        test_amounts=test_amounts,
        threshold=xgb_result["threshold"],
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
        pipeline=xgb_result["pipeline"],
        sample_X=X_test.head(10),
        output_dir=reports_dir,
        dpi=config.get("reporting", {}).get("figure_dpi", 120),
    )

    # ── 8. Error analysis ──────────────────────────────────────────────
    error_analysis(
        y_true=y_test,
        y_prob=xgb_result["test_probs"],
        amounts=test_amounts,
        time_values=test_times,
        threshold=xgb_result["threshold"],
        output_path=os.path.join(results_dir, "error_analysis.json"),
    )

    # ── 9. Threshold sensitivity (validation set) ──────────────────────
    threshold_rows = compare_thresholds(
        y_val,
        xgb_result["val_probs"],
        val_amounts,
        fp_cost,
        fn_multiplier,
    )
    best_row = min(threshold_rows, key=lambda r: r["Total_Cost"])
    logger.info(
        f"Validation cost curve: lowest cost ${best_row['Total_Cost']:.2f} "
        f"at threshold {best_row['Threshold']:.4f}"
    )

    # ── 9b. Evaluation report (validation vs test separation) ─────────────
    report = generate_evaluation_report(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_val=y_val,
        val_probs=xgb_result["val_probs"],
        val_amounts=val_amounts,
        y_test=y_test,
        y_prob=xgb_result["test_probs"],
        test_amounts=test_amounts,
        threshold=xgb_result["threshold"],
        fp_cost=fp_cost,
        fn_multiplier=fn_multiplier,
        comparison=comparison,
        threshold_rows=threshold_rows,
        output_path="reports/evaluation_report.md",
    )
    logger.info("Evaluation report saved to %s", report["report_path"])

    # ── 10. Save final artifacts ────────────────────────────────────────
    # Save the final XGBoost pipeline (best model)
    model_path = os.path.join(artifact_dir, "fraud_pipeline.pkl")
    joblib.dump(xgb_result["pipeline"], model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics (for API to read threshold)
    metrics_output = {
        "model_type": "xgboost",
        "scale_pos_weight": float(scale_pos_weight),
        "class_ratio": {
            "train_negative": int(np.sum(y_train == 0)),
            "train_positive": int(np.sum(y_train == 1)),
            "ratio": float(scale_pos_weight),
        },
        "baseline_model": {
            "type": "logistic_regression",
            "threshold": float(lr_result["threshold"]),
            "metrics": lr_result["metrics"],
        },
        "selected_model": {
            "type": "xgboost",
            "threshold": float(xgb_result["threshold"]),
            "metrics": xgb_result["metrics"],
        },
        "comparison": comparison,
        "tuning": {
            "enabled": tune_config.get("enabled", False) if tune_config else False,
            "best_params": xgb_result.get("best_params"),
            "cv_scoring": tune_config.get("scoring") if tune_config else None,
            "cv_folds": tune_config.get("cv_folds") if tune_config else None,
        },
        "threshold_sensitivity_validation": {
            "best_threshold": float(best_row["Threshold"]),
            "best_cost": float(best_row["Total_Cost"]),
            "selected_threshold": float(xgb_result["threshold"]),
            "cost_at_selected": float(
                compute_metrics(
                    y_val,
                    xgb_result["val_probs"],
                    threshold=xgb_result["threshold"],
                    amounts=val_amounts,
                    fp_cost=fp_cost,
                    fn_multiplier=fn_multiplier,
                )["Total_Cost"]
            ),
            "baseline_threshold": 0.5,
            "baseline_cost": float(
                compute_metrics(
                    y_val,
                    xgb_result["val_probs"],
                    threshold=0.5,
                    amounts=val_amounts,
                    fp_cost=fp_cost,
                    fn_multiplier=fn_multiplier,
                )["Total_Cost"]
            ),
        },
    }

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save comparison results
    comparison_path = os.path.join(results_dir, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Comparison saved to {comparison_path}")

    logger.info("Pipeline completed successfully.")
    logger.info(
        f"\n{'='*60}\n"
        f"FINAL RESULTS (Test Set, Threshold={xgb_result['threshold']:.4f})\n"
        f"{'='*60}\n"
        f"Logistic Regression: PR-AUC={lr_result['metrics']['PR_AUC']:.4f}, "
        f"F1={lr_result['metrics']['F1']:.4f}\n"
        f"XGBoost (weighted):   PR-AUC={xgb_result['metrics']['PR_AUC']:.4f}, "
        f"F1={xgb_result['metrics']['F1']:.4f}\n"
        f"Test cost (XGBoost):  ${xgb_result['metrics']['Total_Cost']:.2f}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
