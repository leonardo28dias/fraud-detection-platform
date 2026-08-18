import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import os

def generate_shap_summary(pipeline: Pipeline, X: pd.DataFrame, output_dir: str):
    """Generates global feature importance explanations using SHAP."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the transformed data and the model from the pipeline
    X_transformed = X.copy()
    for name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)
        
    model = pipeline.named_steps['classifier']
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight')
    plt.close()