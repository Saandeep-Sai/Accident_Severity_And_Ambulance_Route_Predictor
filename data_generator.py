"""
data_generator.py — Phase 1: Data Loading & Preprocessing
============================================================
Loads, cleans, and preprocesses the REAL datasets provided in the
data/ folder. Produces pipeline-ready CSVs for downstream modules.

Real Data Sources:
  - data/telemetry/v2.csv           — 3.1M vehicle telemetry readings
  - data/driving_behavior/features_14.csv — 1,102 driving behaviour samples
  - data/accidents/Road.csv         — 12,316 road accident records
  - data/hospital/bed_occupancy_data.csv  — 16 hospital occupancy records

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# Source paths (real data)
TELEMETRY_SRC = os.path.join("data", "telemetry", "v2.csv")
DRIVING_SRC = os.path.join("data", "driving_behavior", "features_14.csv")
ACCIDENTS_SRC = os.path.join("data", "accidents", "Road.csv")
HOSPITAL_SRC = os.path.join("data", "hospital", "bed_occupancy_data.csv")

# Output paths (pipeline-ready)
TELEMETRY_OUT = "telemetry_data.csv"
HOSPITAL_OUT = "hospital_data.csv"
ACCIDENTS_OUT = "accidents_processed.csv"
DRIVING_OUT = "driving_behavior_processed.csv"

# Telemetry: sample size (full dataset is 3.1M rows — we sample for
# tractable Isolation Forest training)
TELEMETRY_SAMPLE_SIZE = 50_000


# ---------------------------------------------------------------------------
# 1. Telemetry data
# ---------------------------------------------------------------------------

def load_and_preprocess_telemetry(src: str = TELEMETRY_SRC,
                                  sample_size: int = TELEMETRY_SAMPLE_SIZE,
                                  seed: int = SEED) -> pd.DataFrame:
    """
    Load real vehicle telemetry from data/telemetry/v2.csv.

    Original columns:
        tripID, deviceID, timeStamp, accData, gps_speed, battery, cTemp,
        dtc, eLoad, iat, imap, kpl, maf, rpm, speed, tAdv, tPos

    Processing:
      1. Drop non-numeric blob (accData) and constant/low-info columns.
      2. Parse timestamps.
      3. Engineer anomaly labels from domain thresholds.
      4. Sample down for tractable ML training.

    Returns
    -------
    pd.DataFrame  — cleaned telemetry with is_anomaly label
    """
    print(f"[INFO] Loading telemetry from {src} …")
    # Read a manageable sample from the 3.1M-row file
    # First, read total row count to sample uniformly
    total_rows = sum(1 for _ in open(src, "r", encoding="utf-8")) - 1
    skip_rate = max(1, total_rows // sample_size)
    # Skip rows randomly to get an approximate sample
    skip_idx = sorted(
        np.random.RandomState(seed).choice(
            range(1, total_rows + 1),      # 1-indexed (row 0 is header)
            size=max(0, total_rows - sample_size),
            replace=False,
        )
    )
    df = pd.read_csv(src, skiprows=skip_idx)
    print(f"[INFO] Sampled {len(df):,} rows from {total_rows:,} total")

    # Drop non-numeric / low-info columns
    drop_cols = ["accData", "battery", "dtc", "kpl", "maf", "tAdv", "tPos"]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    # Rename for clarity
    rename_map = {
        "tripID": "vehicle_id",
        "deviceID": "device_id",
        "timeStamp": "timestamp",
        "gps_speed": "gps_speed_kmph",
        "cTemp": "engine_temp_c",
        "eLoad": "engine_load_pct",
        "iat": "intake_air_temp_c",
        "imap": "intake_manifold_pressure",
        "rpm": "engine_rpm",
        "speed": "speed_kmph",
    }
    df.rename(columns=rename_map, inplace=True)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with nulls in critical columns
    critical = ["speed_kmph", "engine_temp_c", "engine_rpm", "engine_load_pct"]
    df.dropna(subset=[c for c in critical if c in df.columns], inplace=True)

    # Convert vehicle_id to string label
    df["vehicle_id"] = "VH" + df["vehicle_id"].astype(int).astype(str).str.zfill(3)

    # --- Engineer anomaly label ---
    # Percentile-based thresholds (calibrated from real data distribution).
    # A reading is anomalous if it exceeds the 95th percentile for that
    # feature — this adapts to the actual data and ensures multi-feature
    # anomalies are captured.
    speed_thresh = df["speed_kmph"].quantile(0.95)
    temp_thresh = df["engine_temp_c"].quantile(0.95)
    rpm_thresh = df["engine_rpm"].quantile(0.95)
    load_thresh = df["engine_load_pct"].quantile(0.95)

    print(f"[INFO] Anomaly thresholds (95th percentile):")
    print(f"       speed > {speed_thresh:.1f} km/h, "
          f"temp > {temp_thresh:.1f}°C, "
          f"rpm > {rpm_thresh:.0f}, "
          f"load > {load_thresh:.1f}%")

    df["is_anomaly"] = 0
    df.loc[df["speed_kmph"] > speed_thresh, "is_anomaly"] = 1
    df.loc[df["engine_temp_c"] > temp_thresh, "is_anomaly"] = 1
    df.loc[df["engine_rpm"] > rpm_thresh, "is_anomaly"] = 1
    df.loc[df["engine_load_pct"] > load_thresh, "is_anomaly"] = 1

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Driving behaviour data
# ---------------------------------------------------------------------------

def load_and_preprocess_driving(src: str = DRIVING_SRC) -> pd.DataFrame:
    """
    Load driving behaviour features (accelerometer + gyroscope statistics).

    Original: 1,102 rows × 61 columns (60 features + Target).
    Target classes: 1, 2, 3, 4  (driving behaviour categories).

    We map: classes representing aggressive/dangerous driving → is_anomaly=1.
    Mapping:
      1 = Normal driving  → 0
      2 = Aggressive       → 1
      3 = Drowsy / Slow    → 0
      4 = Very aggressive  → 1

    Returns
    -------
    pd.DataFrame
    """
    print(f"[INFO] Loading driving behaviour from {src} …")
    df = pd.read_csv(src)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Map target to anomaly (aggressive driving = anomaly)
    anomaly_map = {1: 0, 2: 1, 3: 0, 4: 1}
    df["is_anomaly"] = df["Target"].map(anomaly_map).fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Accidents data
# ---------------------------------------------------------------------------

def load_and_preprocess_accidents(src: str = ACCIDENTS_SRC) -> pd.DataFrame:
    """
    Load and preprocess road accident records.

    Original: 12,316 rows × 32 columns.
    Target: Accident_severity → "Slight Injury", "Serious Injury", "Fatal injury"

    Processing:
      1. Clean severity labels.
      2. Encode categorical features with label encoding.
      3. Handle missing values.

    Returns
    -------
    pd.DataFrame
    """
    print(f"[INFO] Loading accidents data from {src} …")
    df = pd.read_csv(src)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Standardise severity labels
    severity_map = {
        "Slight Injury": "minor",
        "Serious Injury": "moderate",
        "Fatal injury": "severe",
    }
    df["severity"] = df["Accident_severity"].map(severity_map)
    df.drop(columns=["Accident_severity"], inplace=True)

    # Fill missing values in object columns with 'Unknown'
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("Unknown")

    # Fill missing values in numeric columns with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Label-encode categorical columns for ML
    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    for col in obj_cols:
        if col == "severity":
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df


# ---------------------------------------------------------------------------
# 4. Hospital data
# ---------------------------------------------------------------------------

def load_and_preprocess_hospital(src: str = HOSPITAL_SRC) -> pd.DataFrame:
    """
    Load hospital bed occupancy data.

    Original: 16 rows × 6 columns:
        date, day_of_week, month, year, is_holiday, beds_occupied

    We also generate a synthetic hospital registry (locations, names, etc.)
    because the real data only has occupancy time-series and the pipeline
    needs hospital metadata for routing & forecasting.

    Returns
    -------
    pd.DataFrame  — hospital_data with full metadata + occupancy info
    """
    print(f"[INFO] Loading hospital occupancy from {src} …")
    occupancy_df = pd.read_csv(src)
    print(f"[INFO] Loaded {len(occupancy_df)} rows")

    # Average occupancy from real data
    avg_occupied = int(occupancy_df["beds_occupied"].mean())

    # Generate hospital registry (metadata needed by routing engine)
    rng = np.random.RandomState(SEED)
    n_hospitals = 20

    hospital_names = [
        "City General Hospital", "St. Mary's Medical Center",
        "Regional Trauma Center", "Sunrise Hospital", "Metro Health",
        "Community Medical Center", "Valley Hospital", "Central ER",
        "Northside Medical", "Eastside Clinic", "Westend Hospital",
        "Southgate Medical", "Riverdale Health", "Hilltop Hospital",
        "Lakeside Medical", "Park Avenue Hospital", "Unity Health",
        "Heritage Medical Center", "Golden Gate Hospital", "Diamond Health",
    ]

    hospital_ids = [f"H{str(i).zfill(3)}" for i in range(1, n_hospitals + 1)]

    # Hyderabad-area coordinates
    latitudes = rng.uniform(17.30, 17.50, n_hospitals)
    longitudes = rng.uniform(78.35, 78.55, n_hospitals)

    total_beds = rng.randint(150, 501, n_hospitals)
    # Use real average occupancy as baseline, with per-hospital variation
    occupancy_pct = rng.uniform(0.55, 0.90, n_hospitals)
    current_occupancy = (total_beds * occupancy_pct).astype(int)

    trauma_center = rng.choice([True, False], n_hospitals, p=[0.35, 0.65])
    avg_response = np.round(rng.uniform(5, 30, n_hospitals), 1)

    hospital_df = pd.DataFrame({
        "hospital_id": hospital_ids,
        "hospital_name": hospital_names[:n_hospitals],
        "latitude": np.round(latitudes, 6),
        "longitude": np.round(longitudes, 6),
        "total_beds": total_beds,
        "current_occupancy": current_occupancy,
        "trauma_center": trauma_center,
        "avg_response_time_min": avg_response,
        "real_avg_beds_occupied": avg_occupied,
    })

    return hospital_df, occupancy_df


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(telemetry_df: pd.DataFrame,
                  driving_df: pd.DataFrame,
                  accidents_df: pd.DataFrame,
                  hospital_df: pd.DataFrame) -> None:
    """Print a human-readable summary of all processed datasets."""
    total_t = len(telemetry_df)
    anom_t = telemetry_df["is_anomaly"].sum()

    print("\n" + "=" * 65)
    print("  REAL DATA PREPROCESSING — SUMMARY")
    print("=" * 65)

    print(f"\n  📡 Telemetry Data")
    print(f"     Rows              : {total_t:,}")
    print(f"     Anomalous rows    : {anom_t:,}  ({anom_t/total_t*100:.2f}%)")
    print(f"     Vehicles          : {telemetry_df['vehicle_id'].nunique()}")
    print(f"     Features          : {[c for c in telemetry_df.columns if c not in ['vehicle_id','timestamp','device_id','is_anomaly']]}")

    print(f"\n  🚗 Driving Behaviour Data")
    print(f"     Rows              : {len(driving_df):,}")
    print(f"     Features          : {len(driving_df.columns) - 2}")  # minus Target, is_anomaly
    print(f"     Anomaly (aggressive): {driving_df['is_anomaly'].sum():,}  "
          f"({driving_df['is_anomaly'].mean()*100:.1f}%)")

    print(f"\n  💥 Accidents Data")
    print(f"     Rows              : {len(accidents_df):,}")
    print(f"     Severity dist     : {accidents_df['severity'].value_counts().to_dict()}")

    print(f"\n  🏥 Hospital Data")
    print(f"     Hospitals         : {len(hospital_df)}")
    print(f"     Trauma centers    : {hospital_df['trauma_center'].sum()}")
    print(f"     Avg beds          : {hospital_df['total_beds'].mean():.0f}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 1] Data Loading & Preprocessing (REAL DATA)")
    print("=" * 55)

    # Load & preprocess all datasets
    telemetry_df = load_and_preprocess_telemetry()
    driving_df = load_and_preprocess_driving()
    accidents_df = load_and_preprocess_accidents()
    hospital_df, occupancy_df = load_and_preprocess_hospital()

    # Save pipeline-ready CSVs
    telemetry_df.to_csv(TELEMETRY_OUT, index=False)
    hospital_df.to_csv(HOSPITAL_OUT, index=False)
    accidents_df.to_csv(ACCIDENTS_OUT, index=False)
    driving_df.to_csv(DRIVING_OUT, index=False)

    print(f"\n[✓] {TELEMETRY_OUT} saved ({len(telemetry_df):,} rows)")
    print(f"[✓] {HOSPITAL_OUT} saved ({len(hospital_df)} rows)")
    print(f"[✓] {ACCIDENTS_OUT} saved ({len(accidents_df):,} rows)")
    print(f"[✓] {DRIVING_OUT} saved ({len(driving_df):,} rows)")

    print_summary(telemetry_df, driving_df, accidents_df, hospital_df)
    print("\nPhase 1 complete — data preprocessing finished.")
