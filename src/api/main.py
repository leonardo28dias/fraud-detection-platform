import json
import logging
import os
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionResponse
from src.data.validation import TransactionSchema

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
PIPELINE_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(MODELS_DIR, "fraud_pipeline.pkl")
)
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(MODELS_DIR, "metrics.json"))


def _load_artifacts() -> tuple:
    """Attempt to load model pipeline and optimal threshold.

    Returns (pipeline, threshold).  If the model file is missing or
    cannot be read, returns (None, 0.5) so the API can still serve
    /health and return 503 on /predict.
    """
    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except (FileNotFoundError, OSError):
        logger.warning("Model artifact not found at %s", PIPELINE_PATH)
        return None, 0.5
    except Exception:
        logger.exception("Failed to load model artifact from %s", PIPELINE_PATH)
        return None, 0.5

    threshold = 0.5
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            threshold = (
                metrics.get("selected_model", {})
                .get("metrics", {})
                .get("Threshold", 0.5)
            )
    except Exception:
        logger.warning("Could not read metrics file; using default threshold 0.5")

    logger.info("Model loaded from %s (threshold=%.4f).", PIPELINE_PATH, threshold)
    return pipeline, threshold


pipeline, optimal_threshold = _load_artifacts()


app = FastAPI(
    title="Fraud Detection Inference API",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "threshold": optimal_threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionSchema):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )
    try:
        df = pd.DataFrame([transaction.model_dump()])
        prob = pipeline.predict_proba(df)[0, 1]
        is_fraud = bool(prob >= optimal_threshold)
        return PredictionResponse(
            transaction_id=str(uuid.uuid4()),
            is_fraud=is_fraud,
            fraud_probability=float(prob),
            applied_threshold=float(optimal_threshold),
        )
    except Exception:
        logger.exception("Prediction failed for transaction")
        raise HTTPException(status_code=500, detail="Prediction failed.")
