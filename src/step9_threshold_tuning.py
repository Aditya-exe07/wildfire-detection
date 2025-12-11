# src/step9_threshold_tuning.py
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

ROOT = Path(".")
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

MODEL_PATH = str(MODELS_DIR / "rf_fire_model.joblib")
SCALER_PATH = str(MODELS_DIR / "scaler_rf.joblib")
TEST_CSV = str(DATA_DIR / "test.csv")

print("Model:", MODEL_PATH)
print("Scaler:", SCALER_PATH)
print("Test CSV:", TEST_CSV)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(TEST_CSV)
FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

# robustly coerce features (same as analysis script)
if "daynight" in df.columns:
    df["daynight"] = df["daynight"].map({"D": 1, "N": 0}).fillna(df["daynight"])
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0
for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

label_col = None
for cand in ["label", "pred_label", "true_label", "target"]:
    if cand in df.columns:
        label_col = cand
        break
if label_col is None:
    raise SystemExit("No label column found in test CSV.")

X = df[FEATURE_COLS].to_numpy(dtype=float)
y_true = df[label_col].astype(int).to_numpy()

X_scaled = scaler.transform(X)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_scaled)[:, 1]
else:
    # fallback: if model doesn't support predict_proba, use decision_function or predictions
    if hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_scaled)
        # scale to [0,1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
    else:
        y_prob = model.predict(X_scaled).astype(float)

# thresholds to try
thresholds = np.concatenate([np.linspace(0.40, 0.90, 26), np.linspace(0.91, 0.99, 9)])
results = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    p, r, f1, supp = precision_recall_fscore_support(y_true, y_pred_t, labels=[1], average=None, zero_division=0)[0], None, None, None
    # use macro/weighted metrics
    p_all, r_all, f1_all, _ = precision_recall_fscore_support(y_true, y_pred_t, average='binary', pos_label=1, zero_division=0)
    tn = int(((y_true == 0) & (y_pred_t == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_t == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_t == 0)).sum())
    tp = int(((y_true == 1) & (y_pred_t == 1)).sum())
    results.append({
        "threshold": float(round(t,3)),
        "precision": float(round(p_all,4)),
        "recall": float(round(r_all,4)),
        "f1": float(round(f1_all,4)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    })

res_df = pd.DataFrame(results).sort_values("threshold", ascending=False)
out_csv = DATA_DIR / "threshold_tuning_results.csv"
res_df.to_csv(out_csv, index=False)
print("Saved threshold results to:", out_csv)
print(res_df[["threshold","precision","recall","f1","tp","fp","fn"]].to_string(index=False))

# ROC & PR curves
roc_auc = roc_auc_score(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(DATA_DIR / "roc_curve.png")
plt.close()
print("Saved ROC curve to data/roc_curve.png")

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(DATA_DIR / "pr_curve.png")
plt.close()
print("Saved PR curve to data/pr_curve.png")

# Save recommended thresholds:
# e.g. threshold that gives recall >= 0.98 with highest precision
candidates = res_df[res_df["recall"] >= 0.98]
recommended = None
if len(candidates) > 0:
    recommended = candidates.sort_values("precision", ascending=False).iloc[0].to_dict()
else:
    # fallback pick threshold with best f1
    recommended = res_df.sort_values("f1", ascending=False).iloc[0].to_dict()

conf_path = DATA_DIR / "threshold_recommendation.json"
with open(conf_path, "w") as f:
    json.dump(recommended, f, indent=2)
print("Wrote threshold recommendation to", conf_path)
print("Recommendation:", recommended)