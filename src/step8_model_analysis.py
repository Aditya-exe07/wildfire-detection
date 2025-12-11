# src/step8_model_analysis.py
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json

ROOT = Path(".")
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Try to find model + scaler
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "rf_fire_model.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", str(MODELS_DIR / "scaler_rf.joblib"))

print("Using model:", MODEL_PATH)
print("Using scaler:", SCALER_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Try to find a test CSV in data/ (common names)
candidates = list(DATA_DIR.glob("*test*.csv")) + list(DATA_DIR.glob("test*.csv")) + list(DATA_DIR.glob("*_test.csv"))
candidates = [p for p in candidates if p.is_file()]
if not candidates:
    # fallback: any CSV with "test" or a small file
    candidates = [p for p in DATA_DIR.glob("*.csv")][:5]

if not candidates:
    raise SystemExit("No test CSV found in data/. Please point TEST_CSV env var to your test file.")

TEST_CSV = os.getenv("TEST_CSV", str(candidates[0]))
print("Using test CSV:", TEST_CSV)

df = pd.read_csv(TEST_CSV)
print("Rows in test file:", len(df))

# Expected feature columns in training
FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

# Check columns present
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print("Warning: missing feature columns in test file:", missing)
    # add zero cols so code runs
    for c in missing:
        df[c] = 0.0

# --- COERCE FEATURES TO NUMERIC (robust) ---
# map daynight if present (D/N -> 1/0)
if "daynight" in df.columns:
    df["daynight"] = df["daynight"].map({"D": 1, "N": 0}).fillna(df["daynight"])

# Now force all FEATURE_COLS to numeric; invalid parsing becomes NaN
for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Fill NaNs with 0 (or consider median/imputation later)
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

# Ensure label column exists
label_col_candidates = ["label", "pred_label", "true_label", "target"]
label_col = None
for cand in label_col_candidates:
    if cand in df.columns:
        label_col = cand
        break

if label_col is None:
    raise SystemExit("No label column found in test CSV. Add a 'label' column (0/1) or set TEST_CSV to proper file.")

y_true = df[label_col].astype(int).to_numpy()
X = df[FEATURE_COLS].to_numpy(dtype=float)

# scale and predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
# if model has predict_proba
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_scaled)[:, 1]
else:
    y_prob = None

# Metrics
print("\n--- CLASSIFICATION REPORT (test) ---")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

# Feature importances (if available)
fi = None
try:
    if hasattr(model, "feature_importances_"):
        fi = list(model.feature_importances_)
        print("\nFeature importances (model.feature_importances_):")
        for c, v in sorted(zip(FEATURE_COLS, fi), key=lambda x: x[1], reverse=True):
            print(f"  {c}: {v:.4f}")
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        print("\nModel coefficients (coef_):")
        for c, v in sorted(zip(FEATURE_COLS, coef), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {c}: {v:.4f}")
    else:
        print("\nNo feature importance/coefficients available for this model type.")
except Exception as e:
    print("Feature importance extraction failed:", e)

# Save misclassified samples for error analysis
out_dir = DATA_DIR
mis_path = out_dir / "misclassified_test_samples.csv"
df_out = df.copy()
df_out["pred"] = y_pred
if y_prob is not None:
    df_out["pred_prob"] = y_prob
df_out["correct"] = (df_out[label_col].astype(int) == df_out["pred"]).astype(int)
mis = df_out[df_out["correct"] == 0]
mis.to_csv(mis_path, index=False)
print(f"\nSaved {len(mis)} misclassified rows to: {mis_path}")

# Save a summary JSON
summary = {
    "test_rows": int(len(df)),
    "misclassified": int(len(mis)),
    "confusion_matrix": cm.tolist(),
    "label_col": label_col,
    "test_csv": TEST_CSV
}
summary_path = out_dir / "model_test_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote summary to", summary_path)