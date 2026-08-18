import yaml
import logging
import joblib
import json
import os
from src.data.ingestion import load_and_split_data
from src.models.trainer import build_pipeline
from src.models.thresholding import optimize_business_threshold
from src.evaluate.metrics import compute_metrics
from src.evaluate.explainability import generate_shap_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['model']['artifact_dir'], exist_ok=True)
    
    # 1. Ingestion & Split
    train_df, val_df, test_df = load_and_split_data(
        filepath=config['data']['raw_path'],
        time_col=config['data']['time_col'],
        val_size=config['data']['val_size'],
        test_size=config['data']['test_size']
    )
    
    X_train = train_df.drop(columns=[config['data']['target_col']])
    y_train = train_df[config['data']['target_col']].values
    
    X_val = val_df.drop(columns=[config['data']['target_col']])
    y_val = val_df[config['data']['target_col']].values
    val_amounts = val_df['Amount'].values
    
    X_test = test_df.drop(columns=[config['data']['target_col']])
    y_test = test_df[config['data']['target_col']].values
    
    # 2. Training
    pipeline = build_pipeline(config['model']['type'], config['model']['params'])
    pipeline.fit(X_train, y_train)
    
    # 3. Threshold Optimization (Using Validation Set)
    logging.info("Optimizing decision threshold based on business costs...")
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    best_thresh = optimize_business_threshold(
        y_true=y_val, 
        y_prob=val_probs, 
        amounts=val_amounts,
        fp_cost=config['business_cost']['false_positive_cost'],
        fn_multiplier=config['business_cost']['false_negative_multiplier']
    )
    logging.info(f"Optimal Threshold: {best_thresh:.4f}")
    
    # 4. Evaluation (Using Test Set)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_probs, threshold=best_thresh)
    logging.info(f"Test Metrics: {metrics}")
    
    # 5. Explainability
    logging.info("Generating SHAP explanations...")
    generate_shap_summary(pipeline, X_test.sample(min(1000, len(X_test)), random_state=42), "reports/")
    
    # 6. Artifact Serialization
    model_path = os.path.join(config['model']['artifact_dir'], "fraud_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    
    metrics_path = os.path.join(config['model']['artifact_dir'], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"test_metrics": metrics}, f, indent=4)
        
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()