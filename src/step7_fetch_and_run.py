# src/step7_fetch_and_run.py
import os
import time
import io
import requests
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes

# clustering
import numpy as np
from sklearn.cluster import DBSCAN
import math

from dotenv import load_dotenv
from pathlib import Path

# Load .env for local development
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# CONFIG - set these environment variables OR edit here
FIRMS_CSV_URL = os.getenv("FIRMS_CSV_URL")  # e.g. "https://firms.modaps.eosdis.nasa.gov/..."
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_fire_model_classweight.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_rf_classweight.joblib")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.44))  # probability threshold to alert
ALERT_EMAIL = os.getenv("ALERT_EMAIL")  # recipient
SMTP_SERVER = os.getenv("SMTP_SERVER")  # e.g. smtp.gmail.com
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")  # app password or SMTP password

# Clustering & dedupe parameters (tune if needed)
KMS_CLUSTER_EPS = float(os.getenv("KMS_CLUSTER_EPS", 10.0))  # cluster radius in kilometers
MIN_CLUSTER_SAMPLES = int(os.getenv("MIN_CLUSTER_SAMPLES", 3))
TOP_N = int(os.getenv("ALERT_TOP_N", 10))
DEDUPE_HOURS = float(os.getenv("DEDUPE_HOURS", 6.0))  # don't re-alert the same area within this many hours

ALERTS_SENT_CSV = OUTPUT_DIR / "alerts_sent.csv"  # stores past alerts to prevent duplicates

# Model loading
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_COLS = [
    "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
    "scan", "track", "daynight", "hour", "month"
]

# Map cache directory for OSM snapshots
MAP_CACHE_DIR = OUTPUT_DIR / "maps"
MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_firms_csv(url: str) -> pd.DataFrame:
    """Fetch FIRMS CSV from a public URL. Returns DataFrame."""
    print(f"[{datetime.utcnow().isoformat()}] Fetching FIRMS CSV from:", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess FIRMS DataFrame to match training features."""
    # ensure columns exist
    if 'acq_time' in df.columns:
        df['acq_time'] = df['acq_time'].astype(str).str.zfill(4)
        df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H%M', errors='coerce')
        df['hour'] = df['acq_datetime'].dt.hour
        df['month'] = df['acq_datetime'].dt.month
    else:
        # if acq_time isn't present, try to use acq_datetime if present
        if 'acq_datetime' in df.columns:
            df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], errors='coerce')
            df['hour'] = df['acq_datetime'].dt.hour
            df['month'] = df['acq_datetime'].dt.month
        else:
            # fallback: set to 0
            df['hour'] = 0
            df['month'] = 0

    # Ensure daynight is present and encoded
    if 'daynight' in df.columns:
        df['daynight'] = df['daynight'].map({'D': 1, 'N': 0}).fillna(df['daynight'])
    else:
        df['daynight'] = 0

    # Keep only the required columns; if missing cols cause KeyError, fill them with zeros
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df_features = df[FEATURE_COLS].copy()
    return df_features, df  # return both for merging metadata

def predict_batch(df_features: pd.DataFrame):
    X = scaler.transform(df_features)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

# ----------------- OSM static map helpers & email sender -----------------

def _osm_static_map_url(lat, lon, zoom=7, width=640, height=320):
    base = "https://staticmap.openstreetmap.de/staticmap.php"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "markers": f"{lat},{lon},lightblue1",
    }
    qs = "&".join(f"{k}={requests.utils.requote_uri(str(v))}" for k, v in params.items())
    return f"{base}?{qs}"

def _cached_map_path(lat, lon, zoom, w, h):
    # normalize lat/lon to 5 decimal places for caching
    return MAP_CACHE_DIR / f"map_{lat:.5f}_{lon:.5f}_z{zoom}_{w}x{h}.png"

def _fetch_and_cache_map(lat, lon, zoom=7, width=640, height=320):
    """
    Return bytes of PNG map, using cache if available.
    """
    p = _cached_map_path(lat, lon, zoom, width, height)
    if p.exists():
        try:
            return p.read_bytes()
        except Exception:
            pass

    url = _osm_static_map_url(lat, lon, zoom=zoom, width=width, height=height)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img_bytes = r.content
        try:
            p.write_bytes(img_bytes)
        except Exception as e:
            print("Warning: failed to write map cache:", e)
        return img_bytes
    except Exception as e:
        print("Warning: failed to fetch OSM static map:", e)
        return None

def send_email_alert(subject: str, body: str, clusters_for_email=None, max_images=3):
    """
    Send HTML email with inline OSM static map images (cached).
    clusters_for_email: list of cluster dicts with keys:
       cluster, mean_lat, mean_lon, count, max_prob, first_time, last_time
    max_images: maximum number of inline images to attach (tune to avoid large emails)
    """
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
        print("SMTP/alert not configured. Skipping email.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL
    msg.set_content(body)

    html_lines = []
    html_lines.append(f"<h3>{subject}</h3>")
    html_lines.append("<div>Summary:</div>")

    attached = 0
    if clusters_for_email:
        for i, cl in enumerate(clusters_for_email, start=1):
            lat, lon = cl["mean_lat"], cl["mean_lon"]
            count = cl.get("count", 1)
            prob = cl.get("max_prob", 0.0)
            ft = cl.get("first_time", "")
            lt = cl.get("last_time", "")

            if attached >= max_images:
                # include link only
                html_lines.append(
                    f"<div style='margin-bottom:8px;'>"
                    f"<strong>Cluster {cl['cluster']}</strong> — count {count} — prob {prob:.3f} "
                    f"<br/><a href='https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}' target='_blank'>Open in OSM</a>"
                    f"</div>"
                )
                continue

            img_bytes = _fetch_and_cache_map(lat, lon, zoom=7, width=640, height=320)
            if img_bytes:
                maintype, subtype = ("image", "png")
                cid = make_msgid(domain="wildguard.local")[1:-1]  # strip <>
                try:
                    # attach inline
                    msg.get_payload()
                    msg.add_related(img_bytes, maintype=maintype, subtype=subtype, cid=cid)
                except Exception:
                    # fallback attachment
                    msg.add_attachment(img_bytes, maintype=maintype, subtype=subtype, filename=f"map_{i}.png")

                html_lines.append(
                    "<div style='margin-bottom:12px'>"
                    f"<h4>Cluster {cl['cluster']} — count {count} — prob {prob:.3f}</h4>"
                    f"<div>Time: {ft} to {lt}</div>"
                    f"<div><a href='https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}' target='_blank'>Open in OSM</a></div>"
                    f"<div style='margin-top:6px'><img src='cid:{cid}' style='max-width:100%;height:auto;border-radius:6px' /></div>"
                    "</div>"
                )
                attached += 1
            else:
                html_lines.append(
                    "<div style='margin-bottom:12px'>"
                    f"<h4>Cluster {cl['cluster']} — count {count} — prob {prob:.3f}</h4>"
                    f"<div>Time: {ft} to {lt}</div>"
                    f"<div><a href='https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}' target='_blank'>Open in OSM</a></div>"
                    "</div>"
                )
    else:
        html_lines.append("<div>No clusters data provided.</div>")

    if clusters_for_email and len(clusters_for_email) > max_images:
        html_lines.append(f"<div style='margin-top:8px;font-size:12px;color:#666;'>Only the top {max_images} cluster maps included. See links for others.</div>")

    html_body = "<html><body>" + "".join(html_lines) + "</body></html>"
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("Alert email with OSM maps sent to", ALERT_EMAIL)
    except Exception as e:
        print("Failed to send alert email:", e)

# ------- helpers for clustering/dedupe -------
def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance between two lat/lon points in kilometers."""
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_sent_alerts():
    """Load alerts_sent.csv (if exists) with columns: timestamp, centroid_lat, centroid_lon"""
    if ALERTS_SENT_CSV.exists():
        try:
            df = pd.read_csv(ALERTS_SENT_CSV, parse_dates=["timestamp"])
            return df
        except Exception:
            return pd.DataFrame(columns=["timestamp","centroid_lat","centroid_lon","max_prob","count"])
    else:
        return pd.DataFrame(columns=["timestamp","centroid_lat","centroid_lon","max_prob","count"])

def append_sent_alert(centroid_lat, centroid_lon, max_prob, count):
    df = load_sent_alerts()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "max_prob": max_prob,
        "count": count
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ALERTS_SENT_CSV, index=False)

def was_recently_alerted(centroid_lat, centroid_lon, hours=DEDUPE_HOURS):
    df = load_sent_alerts()
    if df.empty:
        return False
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return False
    # check if any recent centroid is within kms threshold (use same eps)
    for _, r in recent.iterrows():
        dist = haversine_km(centroid_lat, centroid_lon, float(r["centroid_lat"]), float(r["centroid_lon"]))
        if dist <= KMS_CLUSTER_EPS:
            return True
    return False

# ------- main run_once with clustering & dedupe -------
def run_once():
    if not FIRMS_CSV_URL:
        raise ValueError("Set FIRMS_CSV_URL environment variable to a valid FIRMS CSV download URL")
    df = fetch_firms_csv(FIRMS_CSV_URL)
    features, original = preprocess_df(df)

    preds, probs = predict_batch(features)
    original = original.reset_index(drop=True)
    original['pred_label'] = preds
    original['pred_prob'] = probs

    # Save results with timestamp
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = OUTPUT_DIR / f"predictions_{ts}.csv"
    original.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")

    # append to master log
    master = OUTPUT_DIR / "predictions_master.csv"
    if master.exists():
        original.to_csv(master, index=False, mode='a', header=False)
    else:
        original.to_csv(master, index=False)
    print("Appended to master predictions file.")

    # Alerts: high-probability fires
    high = original[original['pred_prob'] >= ALERT_THRESHOLD].copy()
    if len(high) == 0:
        print("No high-probability events found.")
        return

    print(f"Found {len(high)} high-probability events (prob >= {ALERT_THRESHOLD})")

    # Prepare clustering (DBSCAN with haversine metric – eps in radians)
    coords = high[['latitude','longitude']].to_numpy()
    if len(coords) < 1:
        print("No coords to cluster.")
        return

    eps_rad = KMS_CLUSTER_EPS / 6371.0  # convert km to radians
    clustering = DBSCAN(eps=eps_rad, min_samples=MIN_CLUSTER_SAMPLES, metric='haversine').fit(np.radians(coords))
    high['cluster'] = clustering.labels_

    # Collect clusters (ignore noise)
    clusters = []
    for c in sorted(set(clustering.labels_)):
        if c == -1:  # noise
            continue
        group = high[high['cluster'] == c]
        count = len(group)
        max_prob = float(group['pred_prob'].max())
        mean_lat = float(group['latitude'].mean())
        mean_lon = float(group['longitude'].mean())
        first_time = group['acq_date'].min() if 'acq_date' in group.columns else ''
        last_time = group['acq_date'].max() if 'acq_date' in group.columns else ''
        clusters.append({
            'cluster': int(c),
            'count': int(count),
            'max_prob': max_prob,
            'mean_lat': mean_lat,
            'mean_lon': mean_lon,
            'first_time': first_time,
            'last_time': last_time
        })

    if len(clusters) == 0:
        print("No clusters found after DBSCAN.")
        return

    # Sort clusters by max_prob desc
    clusters = sorted(clusters, key=lambda x: x['max_prob'], reverse=True)
    # Filter clusters by dedupe (only include clusters not alerted recently)
    new_clusters = []
    for cl in clusters:
        centroid_lat = cl['mean_lat']
        centroid_lon = cl['mean_lon']
        if was_recently_alerted(centroid_lat, centroid_lon, hours=DEDUPE_HOURS):
            # skip clusters that were alerted recently
            continue
        new_clusters.append(cl)

    if len(new_clusters) == 0:
        print("No NEW clusters to alert (all clusters were alerted recently).")
        return

    # Build email body for top N new clusters
    topN = min(TOP_N, len(new_clusters))
    body_lines = []
    body_lines.append(f"Top {topN} new wildfire clusters (threshold={ALERT_THRESHOLD}, dedupe_hours={DEDUPE_HOURS}):\n")
    for i, cl in enumerate(new_clusters[:topN], start=1):
        lat, lon = cl['mean_lat'], cl['mean_lon']
        gmap_link = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=10/{lat}/{lon}"
        body_lines.append(
            f"{i}. Cluster {cl['cluster']} — count={cl['count']}, max_prob={cl['max_prob']:.3f}, "
            f"centroid=({lat:.4f},{lon:.4f}), time_range={cl['first_time']} to {cl['last_time']}\n"
            f"   Map: {gmap_link}\n"
        )
        # append to sent alerts log
        append_sent_alert(lat, lon, cl['max_prob'], cl['count'])

    body = "\n".join(body_lines)
    subject = f"[WildGuard] {len(new_clusters)} new high-prob clusters (top {topN})"
    print(subject)
    print(body)

    # send email with inline OSM maps (up to 3 by default)
    send_email_alert(subject, body, clusters_for_email=new_clusters[:topN], max_images=3)

# Scheduler if you want continuous runs
def run_scheduler(interval_minutes: int = 60):
    sched = BlockingScheduler()
    sched.add_job(run_once, "interval", minutes=interval_minutes, next_run_time=datetime.now())
    print(f"Scheduler started: running every {interval_minutes} minutes. CTRL+C to exit.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

if __name__ == "__main__":
    # For development you can run once; for production use run_scheduler()
    run_once()
    # run_scheduler(interval_minutes=60)