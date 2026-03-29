---
title: Proactive AI Emergency Response
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered road accident detection & emergency dispatch
---

# 🚨 Proactive AI Framework for Early Anomaly Assessment & Rapid Emergency Response

A comprehensive AI-powered system for early detection of road vehicle anomalies, accident severity classification, and coordinated emergency response including ambulance dispatch and hospital load forecasting.

**Students:** Anupama Boya, Atukula Saipriya, Nalla Vishishta — CSE-A  
**Guide:** P. Raj Kumar | **Institution:** MLRITM, Hyderabad

---

## 🏗️ Architecture

```
Real Telemetry (3.1M rows) ──► Anomaly Detector (IF + RF) ──► Severity CNN (MobileNetV2)
                                        │                              │
                                        ▼                              ▼
                               Pipeline Controller ──► Routing Engine (Dijkstra)
                                        │                       │
                                        ▼                       ▼
                              Alert System (SMTP) ◄── Hospital Forecaster (RF)
                                        │
                                        ▼
                              Streamlit Dashboard 🚨
```

## 📋 Modules

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `data_generator.py` | Load & preprocess real datasets |
| 2 | `anomaly_detector.py` | Dual-model: Isolation Forest (AUC=0.918) + Random Forest (AUC=1.000) |
| 3 | `severity_classifier.py` | MobileNetV2 CNN — two-stage fine-tuned (74.6% val accuracy) |
| 4 | `routing_engine.py` | Road graph (50 nodes), Dijkstra shortest path, ambulance dispatch |
| 5 | `hospital_forecaster.py` | Random Forest bed availability predictor |
| 6 | `alert_system.py` | SMTP email alerts (mock mode on HF) |
| 7 | `pipeline_controller.py` | Full pipeline orchestration |
| 8 | `dashboard.py` | Interactive Streamlit dashboard |

## 🎯 Dashboard Features

- **KPI Cards** — total incidents, anomaly scores, ambulance ETAs, alert counts
- **Anomaly Trends** — hourly distribution chart + feature breakdown pie chart
- **Road Network Map** — 50-node city graph with hospitals, depots, intersections
- **Dispatch Table** — recent ambulance dispatches with severity, hospital, ETA
- **Model Explainability** — SHAP importance, ROC curves (IF vs RF), CNN training history
- **📸 Image Upload** — upload any accident image for live CNN severity prediction
- **Hospital Overview** — 20 hospitals with bed availability, occupancy, trauma status
- **Run Simulation** — process telemetry rows through the full pipeline in real time

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Random Forest Classifier | AUC-ROC | **1.000** |
| Isolation Forest (unsupervised) | AUC-ROC | **0.918** |
| MobileNetV2 CNN | Val Accuracy | **74.6%** |
| Alert System | Delivery Rate | **100%** |

## 🔧 Local Setup

```bash
# Create venv (Python 3.11.9)
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
.\run_all.ps1                   # PowerShell
# or: .\run_all.bat             # CMD

# Launch dashboard
streamlit run dashboard.py
```

## 📁 Data Sources

| Dataset | Source | Size |
|---------|--------|------|
| Vehicle Telemetry | OBD-II sensors (v2.csv) | 3.1M rows |
| Driving Behavior | Accelerometer/Gyroscope | 1,102 samples |
| Road Accidents | Ethiopian road data | 12,316 records |
| Hospital Occupancy | Bed occupancy time-series | 16 records |
| Severity Images | Accident scene photos | 1,631 images |

## 📝 License

Academic project — MLRITM, Hyderabad © 2026.
