import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_viirs_2024.csv")

# Separate classes
fire_df = df[df["label"] == 1]
nofire_df = df[df["label"] == 0]

print("Fire samples:", len(fire_df))
print("No-fire samples:", len(nofire_df))

# Downsample fire class to match no-fire
fire_downsampled = fire_df.sample(len(nofire_df), random_state=42)

# Combine balanced dataset
balanced_df = pd.concat([fire_downsampled, nofire_df], axis=0).sample(frac=1, random_state=42)

print("Balanced dataset size:", len(balanced_df))

# Split into train/val/test
train_df, temp_df = train_test_split(balanced_df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# Save splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\nSaved:")
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))