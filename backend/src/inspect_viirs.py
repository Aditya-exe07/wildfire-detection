# src/inspect_viirs.py
import pandas as pd
import sys
from pathlib import Path

csv_path = Path("data/viirs-jpss1_2024_United_States.csv")
if not csv_path.exists():
    print("ERROR: CSV not found at", csv_path.resolve())
    sys.exit(1)

df = pd.read_csv(csv_path, low_memory=False)
print("ROWS:", len(df))
print("COLUMNS:", df.columns.tolist())
print("\nNULL COUNTS:")
print(df.isnull().sum())
print("\nDTYPES:")
print(df.dtypes)
print("\nCONFIDENCE VALUE COUNTS:")
if 'confidence' in df.columns:
    print(df['confidence'].value_counts(dropna=False))
else:
    print("No 'confidence' column found.")
# parse datetime if possible
if 'acq_date' in df.columns and 'acq_time' in df.columns:
    df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
    df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H%M', errors='coerce')
    print("\nDatetime parse: nulls =", df['acq_datetime'].isnull().sum())
# latitude/longitude ranges
if 'latitude' in df.columns and 'longitude' in df.columns:
    print("\nLatitude range:", df['latitude'].min(), "to", df['latitude'].max())
    print("Longitude range:", df['longitude'].min(), "to", df['longitude'].max())

# save small sample for quick experiments
sample_path = Path("data/sample_2000_viirs.csv")
sample = df.sample(2000, random_state=42) if len(df) > 2000 else df
sample.to_csv(sample_path, index=False)
print("\nSaved sample to:", sample_path)