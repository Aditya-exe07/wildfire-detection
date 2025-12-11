import pandas as pd
from pathlib import Path

# Load file
path = Path("data/viirs-jpss1_2024_United_States.csv")
df = pd.read_csv(path, low_memory=False)

# Map confidence to binary label
label_map = {"h": 1, "n": 1, "l": 0}
df["label"] = df["confidence"].map(label_map)

print("Label distribution (1 = fire, 0 = no fire):")
print(df["label"].value_counts())

# Clean date/time
df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
df["acq_datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"],
                                    format="%Y-%m-%d %H%M",
                                    errors="coerce")

# Extract useful features
df["hour"] = df["acq_datetime"].dt.hour
df["month"] = df["acq_datetime"].dt.month

# Drop columns we don’t need
drop_cols = ["satellite", "instrument", "version", "type"]
df = df.drop(columns=drop_cols)

# Reorder columns nicely
cols = ["latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
        "scan", "track", "daynight", "hour", "month", "label"]

df_clean = df[cols]

# Save cleaned dataset
out_path = Path("data/cleaned_viirs_2024.csv")
df_clean.to_csv(out_path, index=False)

print("\nSaved cleaned dataset to:", out_path)
print("Rows:", len(df_clean))
print(df_clean.head())