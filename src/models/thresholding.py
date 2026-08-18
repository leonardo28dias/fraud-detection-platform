import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def optimize_business_threshold(y_true: np.ndarray, y_prob: np.ndarray, amounts: np.ndarray, 
                                fp_cost: float, fn_multiplier: float) -> float:
    """
    Finds the probability threshold that minimizes the total expected business cost.
    Cost = (False Positives * fp_cost) + (False Negatives * amounts * fn_multiplier)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_cost = float('inf')
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        # Calculate costs
        fps = np.sum((y_pred == 1) & (y_true == 0))
        fns = np.sum((y_pred == 0) & (y_true == 1))
        
        # Financial impact of false negatives (missed fraud)
        missed_fraud_amounts = amounts[(y_pred == 0) & (y_true == 1)]
        fn_cost = np.sum(missed_fraud_amounts) * fn_multiplier
        
        total_cost = (fps * fp_cost) + fn_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = thresh
            
    return best_thresh