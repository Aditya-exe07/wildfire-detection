# src/step10_retrain_class_weight.py
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import time

ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# Files
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
CLEAN_CSV = DATA_DIR / "cleaned_viirs_2024.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Output
OUT_MODEL = MODELS_DIR / "rf_fire_model_classweight.joblib"
OUT_SCALER = MODELS_DIR / "scaler_rf_classweight.joblib"
SUMMARY_JSON = DATA_DIR / "retrain_summary.json"

FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

def load_data():
    # prefer explicit train/val files if present
    if TRAIN_CSV.exists() and VAL_CSV.exists():
        print("Loading existing train/validation CSVs.")
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV)
        return df_train, df_val
    # else look for cleaned dataset
    if CLEAN_CSV.exists():
        print("Loading cleaned dataset and performing train/val split.")
        df = pd.read_csv(CLEAN_CSV)
        # ensure label column present
        if "label" not in df.columns:
            raise SystemExit("Cleaned dataset missing 'label' column.")
        # stratify split by label
        train, val = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
        return train.reset_index(drop=True), val.reset_index(drop=True)
    # as last fallback, try generic train/test
    if TEST_CSV.exists():
        print("train.csv/validation.csv missing; using test.csv as val and splitting master CSV.")
        df = pd.read_csv(TEST_CSV)
        if "label" not in df.columns:
            raise SystemExit("test.csv missing 'label' column.")
        train, val = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
        return train.reset_index(drop=True), val.reset_index(drop=True)
    raise SystemExit("No training data found. Place data/train.csv & data/validation.csv or data/cleaned_viirs_2024.csv")

def prepare(df):
    # robust coercion like analysis script
    if "daynight" in df.columns:
        df["daynight"] = df["daynight"].map({"D": 1, "N": 0}).fillna(df["daynight"])
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "label" not in df.columns:
        raise SystemExit("Dataframe missing 'label' column required for supervised training.")
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    return X, y

def main():
    start = time.time()
    train_df, val_df = load_data()
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}")

    X_train, y_train = prepare(train_df)
    X_val, y_val = prepare(val_df)

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # baseline model with class_weight balanced
    base_clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)

    # quick RandomizedSearch over a small hyperparam space (fast)
    param_dist = {
        "n_estimators": sp_randint(100, 600),
        "max_depth": sp_randint(6, 40),
        "min_samples_split": sp_randint(2, 10),
        "min_samples_leaf": sp_randint(1, 8),
        "max_features": ["sqrt", "log2", 0.6, 0.8],
        "bootstrap": [True, False]
    }

    # RandomizedSearchCV
    n_iter = 30  # keep reasonable for local dev; increase if you have more time
    rs = RandomizedSearchCV(
        base_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",  # prioritize F1 for class 1
        n_jobs=-1,
        cv=3,
        verbose=2,
        random_state=42
    )

    print("Starting RandomizedSearchCV (this may take a while)...")
    rs.fit(X_train_scaled, y_train)
    print("Randomized search done. Best params:", rs.best_params_)
    best_clf = rs.best_estimator_

    # evaluate on validation
    y_pred = best_clf.predict(X_val_scaled)
    report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    print("\n--- Validation classification report ---")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # save model & scaler
    joblib.dump(best_clf, OUT_MODEL)
    joblib.dump(scaler, OUT_SCALER)
    print("Saved model to:", OUT_MODEL)
    print("Saved scaler to:", OUT_SCALER)

    # write summary json
    elapsed = time.time() - start
    summary = {
        "best_params": rs.best_params_,
        "val_classification_report": report,
        "val_confusion_matrix": cm.tolist(),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "elapsed_seconds": int(elapsed),
        "model_path": str(OUT_MODEL),
        "scaler_path": str(OUT_SCALER)
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary to", SUMMARY_JSON)
    print("Done.")

if __name__ == "__main__":
    main()