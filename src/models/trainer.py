import logging
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

logger = logging.getLogger(__name__)

def build_pipeline(model_type: str, params: Dict[str, Any]) -> Pipeline:
    """Constructs the modeling pipeline."""
    from src.features.engineering import TemporalFeatureExtractor, LogAmountTransformer
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Building pipeline with model: {model_type}")
    
    if model_type == "xgboost":
        classifier = xgb.XGBClassifier(**params)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(steps=[
        ('temporal_features', TemporalFeatureExtractor(time_col='Time')),
        ('amount_transform', LogAmountTransformer(amount_col='Amount')),
        ('scaler', StandardScaler()), # Scaler fitted ONLY on train data to prevent leakage
        ('classifier', classifier)
    ])
    
    return pipeline