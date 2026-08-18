import json
import os
import datetime

def main():
    metrics_path = "models/metrics.json"
    report_path = "reports/business_summary.md"

    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run scripts/run_pipeline.py first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)
    
    metrics = data.get("test_metrics", {})
    
    report_content = f"""# Fraud Detection Pipeline: Business Report
**Generated on:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Pipeline Performance Summary
The model was evaluated on a strictly unseen temporal test set.

* **ROC-AUC:** {metrics.get('ROC_AUC', 0):.4f}
* **PR-AUC:** {metrics.get('PR_AUC', 0):.4f} (Primary metric for severe imbalance)
* **F1-Score:** {metrics.get('F1', 0):.4f}

## 2. Operational Impact
Using the cost-optimized probability threshold of **{metrics.get('Threshold', 0.5):.4f}**:

* **Precision:** {metrics.get('Precision', 0):.2f} (When we flag fraud, we are right {metrics.get('Precision', 0)*100:.0f}% of the time)
* **Recall:** {metrics.get('Recall', 0):.2f} (We successfully catch {metrics.get('Recall', 0)*100:.0f}% of all actual fraud)

### Workload & Missed Fraud
* **False Positives (Wasted Manual Reviews):** {metrics.get('False_Positives', 0)}
* **False Negatives (Missed Fraudulent Transactions):** {metrics.get('False_Negatives', 0)}

## 3. Next Steps
Review the `reports/shap_summary.png` to understand which features are driving these predictions. 
If the business decides manual review capacity has expanded, lower the False Positive Cost in `configs/config.yaml` and rerun the pipeline to catch more fraud at the expense of higher review volume.
"""

    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"Report successfully generated at {report_path}")

if __name__ == "__main__":
    main()