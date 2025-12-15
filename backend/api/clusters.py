# api/clusters.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load .env for local development
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

app = FastAPI()

# Allow Vite dev server origins for development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALERTS_FILE = Path("data/alerts_sent.csv")

def read_alerts(limit=100):
    """Read alerts_sent.csv and return a list of cluster dicts for the frontend."""
    if not ALERTS_FILE.exists():
        return []
    # Read CSV, parse timestamp as tz-aware UTC
    df = pd.read_csv(ALERTS_FILE, parse_dates=["timestamp"], dayfirst=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # keep recent (last 7 days)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    if "timestamp" in df.columns:
        df = df[df["timestamp"] >= cutoff]
    # sort newest first and limit
    df = df.sort_values("timestamp", ascending=False).head(limit)
    out = []
    for idx, r in df.iterrows():
        try:
            out.append({
                "cluster": int(r.get("cluster", int(idx))),
                "count": int(r.get("count", 1)),
                "max_prob": float(r.get("max_prob", 0.0)),
                "mean_lat": float(r.get("centroid_lat")),
                "mean_lon": float(r.get("centroid_lon")),
                "first_time": r.get("timestamp").isoformat() if pd.notna(r.get("timestamp")) else None,
                "last_time": r.get("timestamp").isoformat() if pd.notna(r.get("timestamp")) else None,
            })
        except Exception:
            # skip malformed rows
            continue
    return out

@app.get("/api/clusters")
def get_clusters(limit: int = 50):
    """Return recent clusters as JSON for the frontend dashboard."""
    return read_alerts(limit=limit)