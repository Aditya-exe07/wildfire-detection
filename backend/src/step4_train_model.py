import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

# Convert daynight (D/N) → numeric
train["daynight"] = train["daynight"].map({"D": 1, "N": 0})
val["daynight"] = val["daynight"].map({"D": 1, "N": 0})

# Split features and labels
X_train = train.drop("label", axis=1)
y_train = train["label"]

X_val = val.drop("label", axis=1)
y_val = val["label"]

# Normalize numeric columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
import joblib
joblib.dump(scaler, "models/scaler.pkl")

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# Save model
model.save("models/fire_detection_model.h5")

print("\nModel training complete.")
print("Saved model to models/fire_detection_model.h5")