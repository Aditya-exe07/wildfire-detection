# WildGuard – AI Wildfire Detection System
### AI-Powered Real-Time Wildfire Detection & Alerting System

WildGuard is an end-to-end machine learning system that detects, clusters, and monitors wildfires in near real-time using satellite imagery from **NASA FIRMS (VIIRS)**.  
The system combines data engineering, machine learning, geospatial clustering, alert automation, and an interactive web dashboard.

## System Architecture

WildGuard is designed as a modular, scalable system with clear separation of concerns.

### 1. Data Source
- **NASA FIRMS (VIIRS)** near–real-time satellite fire detections
- Global coverage, updated multiple times per day

### 2. ML & Analytics Backend
- Feature engineering from satellite observations
- Random Forest classifier for wildfire probability prediction
- Threshold tuning to balance precision vs recall
- DBSCAN-based geospatial clustering to group nearby fire detections
- De-duplication logic to avoid repeated alerts for the same region

### 3. Alerting Pipeline
- High-confidence wildfire clusters trigger alerts
- Automated email notifications with:
  - Cluster confidence
  - Location centroid
  - OpenStreetMap-based static map preview

### 4. API Layer
- FastAPI backend serving clustered wildfire data
- REST endpoint consumed by the frontend dashboard

### 5. Frontend Dashboard
- Interactive map-based visualization (Leaflet + React)
- Real-time cluster updates
- Timeline and probability-based visual cues

## Technology Stack & Rationale

### Machine Learning
- **Scikit-learn (Random Forest)**
  - Chosen for strong performance on tabular satellite data
  - Handles non-linear feature interactions well
  - Interpretable via feature importance analysis

- **DBSCAN (Geospatial Clustering)**
  - Automatically discovers wildfire clusters without preset counts
  - Robust to noise (isolated false detections)
  - Well-suited for spatial density-based grouping

### Backend
- **Python**
  - Mature ML ecosystem
  - Strong data processing libraries

- **FastAPI**
  - High-performance async API
  - Automatic OpenAPI documentation
  - Clean separation between ML logic and API layer

### Frontend
- **React**
  - Component-based architecture
  - Easy integration with real-time APIs

- **Leaflet + OpenStreetMap**
  - Open-source mapping (no vendor lock-in)
  - Ideal for geospatial visualization

- **Tailwind CSS**
  - Rapid UI development
  - Consistent design without heavy CSS overhead

### Infrastructure & DevOps
- **Environment Variables**
  - Secure handling of API keys, SMTP credentials, and thresholds
  - Deployment-ready (GitHub, HF Spaces, cloud platforms)

- **GitHub**
  - Version control
  - Reproducibility
  - Public portfolio visibility

  ## What Makes WildGuard Stand Out

### 1. End-to-End Real-World System
WildGuard is not just a trained model. It is a complete production-style system that:
- Ingests live satellite data (NASA FIRMS)
- Performs ML inference in near real-time
- Aggregates predictions spatially using clustering
- Triggers automated alerts
- Visualizes results on an interactive map dashboard

This mirrors how real-world AI systems are deployed in disaster response.

### 2. Decision-Oriented Machine Learning
Instead of optimizing only for accuracy:
- Precision–recall tradeoffs were explicitly analyzed
- Probability thresholds were tuned for operational use
- False positives were minimized to prevent alert fatigue

This reflects real deployment constraints rather than academic benchmarks.

### 3. Spatiotemporal Intelligence
Wildfires are not independent points.
WildGuard models fires as:
- **Spatial clusters** (using DBSCAN)
- **Temporal events** (with deduplication windows)
- **Evolving phenomena**, not isolated predictions

This adds a layer of reasoning beyond raw classification.

### 4. Interpretability & Model Analysis
The system includes:
- Feature importance analysis
- Misclassification inspection
- Threshold sensitivity studies

These steps demonstrate responsible AI practices and model understanding.

### 5. Deployment-Ready Architecture
- Environment-variable based configuration
- Modular backend/frontend separation
- Easily deployable to cloud platforms or HF Spaces

The project was built with reproducibility and scalability in mind.

## Research Contribution & Academic Relevance

WildGuard explores the intersection of **machine learning, geospatial analytics, and real-time decision systems**.  
Potential academic contributions include:

### 1. Threshold-Aware Disaster Detection
The project demonstrates how wildfire detection performance changes under different probability thresholds, highlighting the tradeoff between:
- Early detection
- False alert suppression
- Operational feasibility

This aligns with research in risk-sensitive and cost-aware ML.

### 2. Spatiotemporal Aggregation of Predictions
Rather than treating satellite detections independently, WildGuard introduces:
- Density-based spatial clustering
- Time-window based deduplication

This approach reduces noise and converts raw predictions into actionable events.

### 3. Model Behavior Analysis at Scale
The system includes:
- Feature importance analysis
- Misclassification case studies
- Class imbalance mitigation through weighted retraining

These analyses provide insight into model reliability and failure modes.

### 4. Reproducible ML Pipeline
All experiments (training, evaluation, threshold tuning) are scripted and reproducible, enabling:
- Future extensions
- Comparative studies
- Dataset or model substitutions

This structure supports academic validation and experimentation.

WildGuard can serve as a foundation for a research paper in:
- Applied Machine Learning
- Geospatial AI
- AI for Social Good / Disaster Response

## How to Run (Local Setup)

This section explains how to run the **WildGuard** wildfire detection system locally.

---

### 1. Clone the Repository
```bash
git clone https://github.com/Aditya-exe07/wildfire-detection.git
cd wildfire-detection
```
### 2.Set up Environment Variables
Create .env file from the template:
```bash
cp .env.example .env
```
Edit .env and fill in:
NASA FIRMS API URL
Email credentials (SMTP)
Alert thresholds
Model paths

### 3. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the API Server
```bash
uvicorn api.main:app --reload
```
The API will be available at:
http://localhost:8000

### 5. Run Wildfire Detection & Alerts
```bash
python src/step7_fetch_and_run.py
```
### 6. Optional: Model Evaluation and Analysis
Run these only if you want to improve the model
```bash
python src/step8_model_analysis.py
python src/step9_threshold_tuning.py
python src/step10_retrain_class_weight.py
```

### Output
Predictions saved to data/
Cluster summaries exposed via API
Email alerts sent with map links
Evaluation reports saved as CSV/JSON

### Tech Stack
Backend: FastAPI, Python
ML: Scikit-learn (Random Forest)
Data: NASA FIRMS (VIIRS)
Maps: OpenStreetMap
Frontend: React + Leaflet
