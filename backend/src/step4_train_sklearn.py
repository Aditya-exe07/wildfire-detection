# src/step4_train_sklearn.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import joblib

# Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# Convert daynight column to numeric (just like in TF)
train["daynight"] = train["daynight"].map({"D": 1, "N": 0})
val["daynight"] = val["daynight"].map({"D": 1, "N": 0})
test["daynight"] = test["daynight"].map({"D": 1, "N": 0})

# Features & labels
X_train = train.drop("label", axis=1)
y_train = train["label"]

X_val = val.drop("label", axis=1)
y_val = val["label"]

X_test = test.drop("label", axis=1)
y_test = test["label"]

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Train RandomForest model
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_s, y_train)

# Evaluate model
print("\n--- VALIDATION SET REPORT ---")
y_pred_val = clf.predict(X_val_s)
print(classification_report(y_val, y_pred_val))

print("\n--- TEST SET REPORT ---")
y_pred_test = clf.predict(X_test_s)
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test))

# Save model & scaler
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/rf_fire_model.joblib")
joblib.dump(scaler, "models/scaler_rf.joblib")

print("\nModel and scaler saved successfully!")