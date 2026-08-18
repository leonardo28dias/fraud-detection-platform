# End-to-End Fraud Detection Platform

## Overview
This repository contains a production-ready machine learning platform designed to detect fraudulent credit card transactions. The project was built to demonstrate a complete, end-to-end data lifecycle—bridging the gap between exploratory data science and backend software engineering. 

Rather than stopping at a Jupyter Notebook, this project implements a reproducible data pipeline, aligns model evaluation with actual business costs, and deploys the resulting model via a REST API.

## Technical Architecture
* **Machine Learning:** XGBoost, Scikit-Learn, SHAP, Pandas
* **Backend API:** FastAPI, Uvicorn
* **Environment & Dependency Management:** Poetry
* **Language:** Python 3.12

## Key Engineering Decisions
* **Business-Driven Metric Optimization:** Default ML models often use a 0.5 decision threshold. In fraud detection, the cost of a False Negative (missing actual fraud) heavily outweighs a False Positive (declining a valid transaction). This pipeline dynamically calculates and optimizes the decision boundary (settling at `0.0991`), directly aligning mathematical outputs with business risk.
* **Temporal Data Splitting:** To prevent data leakage and simulate real-world production environments, the data pipeline uses chronological train/validation/test splits rather than random sampling.
* **Explainability (XAI):** Integrated SHAP (SHapley Additive exPlanations) to generate feature importance summaries, ensuring the model's decisions remain transparent for auditing and compliance purposes.
* **Strict Dependency Resolution:** Utilized Poetry to guarantee reproducible builds across machines, explicitly navigating complex versioning constraints between compiled C-extensions (e.g., NumPy 2.0 breaking changes).

## AI Pair-Programming & Continuous Learning
In building this platform, I utilized LLMs as a pair-programming partner. I architected the system, defined the business logic, and managed the deployment structure. However, I want to be transparent with prospective employers that AI was instrumental in unblocking specific, low-level technical hurdles. 

There are sections of this codebase that I accepted to maintain project momentum and am currently studying to fully understand:
* **XGBoost and SHAP Interoperability:** Navigating the internal binary serialization formats between XGBoost 2.x and SHAP `TreeExplainer`, specifically regarding how SHAP parses XGBoost's raw memory buffers.
* **Pipeline Estimator Extraction:** The precise mechanics of extracting underlying booster objects from Scikit-Learn `Pipeline` wrappers to satisfy specific library constraints.
* **Compiled C-Extension Conflicts:** The cascade effect of the NumPy 2.0 release on older, pre-compiled modules (such as the removal of `np.obj2sctype`), and how to navigate dependency resolution to maintain stability.

This approach allowed me to build a more ambitious, production-like system while providing a clear roadmap for my ongoing technical development.

## Model Performance
Evaluated on a heavily imbalanced transaction dataset (kaggle's creditcard.csv), the model prioritizes Precision-Recall AUC (PR-AUC) over standard ROC-AUC to accurately reflect performance on the minority (fraud) class.

* **ROC-AUC:** 0.9767
* **PR-AUC:** 0.7532
* **F1-Score:** 0.6724

**Test Set Impact:** By utilizing the optimized `0.0991` threshold, the model successfully caught 75% of actual fraud cases while restricting False Positives to just 25 transactions out of over 42,000.

## Reproduction & Setup

### 1. Repository Setup
```bash
git clone [https://github.com/your-username/fraud-detection-platform.git](https://github.com/your-username/fraud-detection-platform.git)
cd fraud-detection-platform
```

### 2. Environment Initialization
This project relies on Poetry for deterministic dependency management.
```bash
poetry install
```

### 3. Pipeline Execution
Run the complete ML pipeline (Ingestion -> Split -> Train -> Optimize -> Evaluate).
Note: Ensure dataset is placed in the data/raw/ directory. This file is excluded via .gitignore due to size.

```bash
poetry run python scripts/run_pipeline.py
```

### 4. 4. API Deployment

Start the local inference server.

```bash
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to http://localhost:8000/docs to interact with the API via the Swagger UI.

### Leonardo Dias
https://www.linkedin.com/in/leonardodias-dev/
