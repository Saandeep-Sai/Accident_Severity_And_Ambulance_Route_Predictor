"""
pipeline_controller.py — Phase 7: Pipeline Controller (Integration)
=====================================================================
Orchestrates the full proactive AI emergency response pipeline:
  1. Anomaly detection (Isolation Forest)
  2. Severity classification (MobileNetV2 CNN)
  3. Hospital load forecasting (Random Forest)
  4. Ambulance routing (NetworkX Dijkstra)
  5. Alert system (SMTP / mock)

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import csv
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Import project modules ---
import anomaly_detector
import severity_classifier
import routing_engine
import hospital_forecaster
import alert_system

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TELEMETRY_CSV = "telemetry_data.csv"
HOSPITAL_CSV = "hospital_data.csv"
GRAPH_PATH = "road_network.gpickle"
INCIDENT_LOG = "incident_log.csv"

# Validation image directory (used for severity prediction)
VAL_IMG_DIR = os.path.join("data", "images", "validation")


# ---------------------------------------------------------------------------
# Incident schema
# ---------------------------------------------------------------------------

@dataclass
class IncidentRecord:
    """Central schema for a detected incident."""
    vehicle_id: str = ""
    timestamp: str = ""
    telemetry_row: dict = field(default_factory=dict)
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    severity: str = "none"
    assigned_ambulance: str = ""
    assigned_hospital: str = ""
    eta_min: float = 0.0
    alert_sent: bool = False
    top_features: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------

_graph = None
_hospital_df = None
_ambulances = None
_severity_images = {}  # cache of image paths per severity class


def _init_pipeline():
    """Load all models and data required by the pipeline."""
    global _graph, _hospital_df, _ambulances, _severity_images

    print("[Pipeline] Initialising modules …")

    # Load anomaly detector (lazy-loaded on first detect_anomaly call)
    anomaly_detector._ensure_loaded()
    print("  ✓ Anomaly detector loaded")

    # Load severity model (lazy-loaded on first predict_severity call)
    severity_classifier._ensure_model_loaded()
    print("  ✓ Severity classifier loaded")

    # Load hospital forecaster
    hospital_forecaster._ensure_loaded()
    print("  ✓ Hospital forecaster loaded")

    # Load road network graph
    _graph = routing_engine.load_graph(GRAPH_PATH)
    print(f"  ✓ Road network loaded ({_graph.number_of_nodes()} nodes)")

    # Load hospital data
    _hospital_df = pd.read_csv(HOSPITAL_CSV)
    print(f"  ✓ Hospital data loaded ({len(_hospital_df)} hospitals)")

    # Initialise ambulance fleet at depot nodes
    depot_nodes = [
        n for n, d in _graph.nodes(data=True)
        if d.get("node_type") == "depot"
    ]
    _ambulances = [
        {"ambulance_id": f"AMB{i+1:02d}", "current_node": depot_nodes[i]}
        for i in range(len(depot_nodes))
    ]
    print(f"  ✓ {len(_ambulances)} ambulances ready")

    # Cache severity image paths for prediction
    for sev in ["minor", "moderate", "severe"]:
        sev_dir = os.path.join(VAL_IMG_DIR, sev)
        if os.path.isdir(sev_dir):
            files = [os.path.join(sev_dir, f) for f in os.listdir(sev_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            _severity_images[sev] = files
    total_imgs = sum(len(v) for v in _severity_images.values())
    print(f"  ✓ {total_imgs} severity images cached for prediction")

    print("[Pipeline] Initialisation complete.\n")


def _get_random_severity_image() -> str:
    """Pick a random severity image from the validation set."""
    all_imgs = []
    for imgs in _severity_images.values():
        all_imgs.extend(imgs)
    if not all_imgs:
        return ""
    return np.random.choice(all_imgs)


def _init_incident_log():
    """Create incident log CSV with headers if it doesn't exist."""
    if not os.path.exists(INCIDENT_LOG):
        with open(INCIDENT_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "vehicle_id", "is_anomaly", "anomaly_score",
                "severity", "assigned_ambulance", "assigned_hospital",
                "eta_min", "alert_sent", "top_features",
            ])


def _log_incident(record: IncidentRecord):
    """Append an incident to the log CSV."""
    _init_incident_log()
    with open(INCIDENT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            record.timestamp,
            record.vehicle_id,
            record.is_anomaly,
            record.anomaly_score,
            record.severity,
            record.assigned_ambulance,
            record.assigned_hospital,
            record.eta_min,
            record.alert_sent,
            "|".join(record.top_features),
        ])


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def process_telemetry_row(row: dict) -> IncidentRecord:
    """
    Process a single telemetry row through the full pipeline.

    Steps:
      1. Run anomaly detection → is_anomaly, anomaly_score
      2. If normal → return early
      3. If anomaly → run severity classification (CNN)
      4. Forecast hospital load for top 3 hospitals
      5. Dispatch nearest ambulance
      6. Send alert
      7. Log incident

    Parameters
    ----------
    row : dict
        Telemetry row with feature columns.

    Returns
    -------
    IncidentRecord
    """
    record = IncidentRecord(
        vehicle_id=row.get("vehicle_id", "Unknown"),
        timestamp=datetime.now().isoformat(),
        telemetry_row=row,
    )

    # Step 1: Anomaly detection
    detection = anomaly_detector.detect_anomaly(row)
    record.is_anomaly = detection["is_anomaly"]
    record.anomaly_score = detection["anomaly_score"]
    record.top_features = detection["top_features"]

    # Step 2: If not anomalous, return early
    if not record.is_anomaly:
        return record

    # Step 3: Severity classification
    img_path = _get_random_severity_image()
    if img_path:
        sev_result = severity_classifier.predict_severity(img_path)
        record.severity = sev_result["severity"]
    else:
        record.severity = "moderate"  # default if no images available

    # Step 4: Hospital load forecasting for top hospitals
    # Pick a random intersection as incident location
    intersection_nodes = [
        n for n, d in _graph.nodes(data=True)
        if d.get("node_type") == "intersection"
    ]
    incident_node = np.random.choice(intersection_nodes)

    top_hospitals = routing_engine.find_nearest_hospital(
        _graph, incident_node, _hospital_df, top_n=3
    )

    # Enrich with forecasted load
    now = datetime.now()
    for h in top_hospitals:
        forecast = hospital_forecaster.forecast_hospital_load(
            h["hospital_id"], now.hour, now.weekday()
        )
        h["load_status"] = forecast["load_status"]
        h["forecasted_beds"] = forecast["predicted_available_beds"]

    # Pick best hospital (first by score, already sorted)
    if top_hospitals:
        best_hospital = top_hospitals[0]
        record.assigned_hospital = best_hospital["hospital_name"]
    else:
        record.assigned_hospital = "Unknown"

    # Step 5: Dispatch ambulance
    dispatch = routing_engine.assign_ambulance(
        incident_node, _ambulances, _graph
    )
    record.assigned_ambulance = dispatch["ambulance_id"]
    record.eta_min = dispatch["eta_min"]

    # Step 6: Send alert
    incident_info = {
        "vehicle_id": record.vehicle_id,
        "location": f"Node {incident_node}",
        "severity": record.severity,
        "anomaly_type": "|".join(record.top_features[:2]),
        "timestamp": record.timestamp,
    }
    alert_result = alert_system.trigger_alert(incident_info)
    record.alert_sent = alert_result["email_sent"] or alert_result["sms_sent"]

    # Step 7: Log
    _log_incident(record)

    return record


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(n_rows: int = 50) -> dict:
    """
    Run the pipeline on a random sample of telemetry rows.

    Parameters
    ----------
    n_rows : int
        Number of rows to process.

    Returns
    -------
    dict  — simulation summary
    """
    _init_pipeline()

    # Load telemetry and sample
    df = pd.read_csv(TELEMETRY_CSV)
    sample = df.sample(n=min(n_rows, len(df)), random_state=42)

    print(f"[Simulation] Processing {len(sample)} telemetry rows …\n")

    total = 0
    anomalies = 0
    alerts_sent = 0
    etas = []

    for _, row in sample.iterrows():
        record = process_telemetry_row(row.to_dict())
        total += 1

        if record.is_anomaly:
            anomalies += 1
            if record.alert_sent:
                alerts_sent += 1
            etas.append(record.eta_min)

    avg_eta = np.mean(etas) if etas else 0.0

    summary = {
        "total_processed": total,
        "anomalies_detected": anomalies,
        "alerts_sent": alerts_sent,
        "avg_eta_min": round(avg_eta, 2),
    }

    print("\n" + "=" * 55)
    print("  SIMULATION SUMMARY")
    print("=" * 55)
    print(f"  Rows processed      : {summary['total_processed']}")
    print(f"  Anomalies detected  : {summary['anomalies_detected']}")
    print(f"  Alerts sent         : {summary['alerts_sent']}")
    print(f"  Avg ambulance ETA   : {summary['avg_eta_min']} min")
    print("=" * 55)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 7] Pipeline Controller — Full Integration")
    print("=" * 55)

    # Remove old incident log to start fresh
    if os.path.exists(INCIDENT_LOG):
        os.remove(INCIDENT_LOG)

    summary = run_simulation(50)
    print("\nPipeline controller ready.")
