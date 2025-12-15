# src/step5_inference.py
import numpy as np
import pandas as pd
import joblib
from typing import Union, List, Dict

# Load model and scaler
model = joblib.load("models/rf_fire_model.joblib")
scaler = joblib.load("models/scaler_rf.joblib")

# Required feature order
FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

def preprocess_input(data: Union[Dict, List[Dict]]) -> np.ndarray:
    """
    Accepts single dict or list of dicts.
    Ensures correct feature order and scaling.
    Returns scaled numpy array ready for model.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])  # convert single dict to 1-row DF
    else:
        df = pd.DataFrame(data)

    # Encode daynight if in D/N format
    if "daynight" in df.columns:
        df["daynight"] = df["daynight"].map({"D": 1, "N": 0}).fillna(df["daynight"])

    # Reorder columns
    df = df[FEATURE_COLS]

    # Scale numeric values
    scaled = scaler.transform(df)

    return scaled


def predict_fire(data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """
    Returns prediction + probabilities.
    """
    X = preprocess_input(data)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # probability fire = 1

    results = []

    if isinstance(data, dict):
        return {
            "prediction": int(preds[0]),
            "probability_fire": float(probs[0])
        }

    # Batch case
    for pred, prob in zip(preds, probs):
        results.append({
            "prediction": int(pred),
            "probability_fire": float(prob)
        })

    return results


# Demo usage
if __name__ == "__main__":
    sample_point = {
        "latitude": 32.1,
        "longitude": -118.5,
        "bright_ti4": 312.0,
        "bright_ti5": 276.0,
        "frp": 2.5,
        "scan": 0.4,
        "track": 0.6,
        "daynight": "N",
        "hour": 6,
        "month": 7
    }

    print("\nSingle Prediction:")
    print(predict_fire(sample_point))

    sample_batch = [sample_point, sample_point]
    print("\nBatch Prediction:")
    print(predict_fire(sample_batch))