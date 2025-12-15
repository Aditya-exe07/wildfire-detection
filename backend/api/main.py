# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Load .env for local development
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    
# Load model and scaler
model = joblib.load("models/rf_fire_model.joblib")
scaler = joblib.load("models/scaler_rf.joblib")

FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

# Input schemas
class FirePoint(BaseModel):
    latitude: float
    longitude: float
    bright_ti4: float
    bright_ti5: float
    frp: float
    scan: float
    track: float
    daynight: str  # "D" or "N"
    hour: int
    month: int

class FireBatch(BaseModel):
    points: List[FirePoint]


app = FastAPI(title="Wildfire Detection API", version="1.0")


def preprocess(df: pd.DataFrame) -> np.ndarray:
    df["daynight"] = df["daynight"].map({"D": 1, "N": 0})
    df = df[FEATURE_COLS]
    return scaler.transform(df)


@app.post("/predict_single")
def predict_single(point: FirePoint):
    df = pd.DataFrame([point.dict()])
    X = preprocess(df)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "probability_fire": prob}


@app.post("/predict_batch")
def predict_batch(batch: FireBatch):
    df = pd.DataFrame([p.dict() for p in batch.points])
    X = preprocess(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        results.append({
            "prediction": int(pred),
            "probability_fire": float(prob)
        })
    return results