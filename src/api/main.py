import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictionResponse
from src.data.validation import TransactionSchema

app = FastAPI(title="Fraud Detection Inference API", version="1.0.0")

# Load model and config on startup
pipeline = None
optimal_threshold = 0.5

@app.on_event("startup")
def load_artifacts():
    global pipeline, optimal_threshold
    try:
        pipeline = joblib.load("models/fraud_pipeline.pkl")
        
        # Load threshold from evaluation metrics (fallback to 0.5)
        import json
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
            optimal_threshold = metrics.get("test_metrics", {}).get("Threshold", 0.5)
    except Exception as e:
        print(f"Warning: Could not load artifacts. {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionSchema):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction.model_dump()])
        
        # Predict
        prob = pipeline.predict_proba(df)[0, 1]
        is_fraud = bool(prob >= optimal_threshold)
        
        return PredictionResponse(
            transaction_id=str(uuid.uuid4()),
            is_fraud=is_fraud,
            fraud_probability=float(prob),
            applied_threshold=optimal_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))