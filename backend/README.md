# WildGuard Backend

This folder contains the backend services and ML pipeline for **WildGuard**, an AI-powered wildfire detection system using NASA FIRMS data.

## Components

### API
- `api/main.py` – FastAPI application entry point
- `api/clusters.py` – Cluster aggregation endpoint used by the frontend map

### ML Pipeline
- `src/step2_clean_and_label.py` – Data cleaning & labeling
- `src/step3_balance_and_split.py` – Dataset balancing and splits
- `src/step4_train_model.py` – Model training
- `src/step8_model_analysis.py` – Evaluation & error analysis
- `src/step9_threshold_tuning.py` – Precision/recall threshold tuning
- `src/step10_retrain_class_weight.py` – Retraining with class weights

### Alerting
- `src/step7_fetch_and_run.py` – Fetches FIRMS data, runs inference, clusters fires, sends email alerts

## Environment Variables
All sensitive values are loaded via environment variables (see `.env.example`).

## Run API (local)
```bash
uvicorn api.main:app --reload
```
