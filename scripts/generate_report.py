import datetime
import json
import os


def main():
    metrics_path = "models/metrics.json"
    report_path = "reports/business_summary.md"

    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run scripts/run_pipeline.py first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    # Test-set metrics for the selected model live under
    # data["selected_model"]["metrics"] (see scripts/run_pipeline.py).
    selected = data.get("selected_model", {})
    metrics = selected.get("metrics", {})
    model_type = selected.get("type", "model")

    roc_auc = metrics.get("ROC_AUC", 0)
    pr_auc = metrics.get("PR_AUC", 0)
    f1 = metrics.get("F1", 0)
    threshold = metrics.get("Threshold", 0.5)
    precision = metrics.get("Precision", 0)
    recall = metrics.get("Recall", 0)
    false_positives = metrics.get("False_Positives", 0)
    false_negatives = metrics.get("False_Negatives", 0)

    report_content = f"""# Fraud Detection Pipeline: Business Report
**Generated on:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Pipeline Performance Summary
Selected model: **{model_type}**, evaluated on a strictly unseen temporal test set.

* **ROC-AUC:** {roc_auc:.4f}
* **PR-AUC:** {pr_auc:.4f} (Primary metric for severe imbalance)
* **F1-Score:** {f1:.4f}

## 2. Operational Impact
Using the cost-optimized probability threshold of **{threshold:.4f}**:

* **Precision:** {precision:.2f} (we are right {precision * 100:.0f}% of the time)
* **Recall:** {recall:.2f} (we catch {recall * 100:.0f}% of all actual fraud)

### Workload & Missed Fraud
* **False Positives (Wasted Manual Reviews):** {false_positives}
* **False Negatives (Missed Fraudulent Transactions):** {false_negatives}

## 3. Next Steps
Review the `reports/shap_summary.png` to understand which features drive predictions.
If the business decides manual review capacity has expanded, lower the False Positive
Cost in `configs/config.yaml` and rerun the pipeline to catch more fraud at the expense
of higher review volume.
"""

    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Report successfully generated at {report_path}")


if __name__ == "__main__":
    main()
