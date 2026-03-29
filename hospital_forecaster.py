"""
hospital_forecaster.py — Phase 5: Hospital Load Forecasting
==============================================================
Trains a Random Forest Regressor to predict hospital bed availability
using the REAL bed_occupancy_data.csv as a baseline, augmented with
synthetic temporal features for training.

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
MODEL_PATH = "hospital_forecaster.pkl"
HOSPITAL_CSV = "hospital_data.csv"
BED_OCCUPANCY_CSV = os.path.join("data", "hospital", "bed_occupancy_data.csv")


def load_real_occupancy(csv_path: str = BED_OCCUPANCY_CSV) -> pd.DataFrame:
    """
    Load the real hospital bed occupancy time-series.

    Columns: date, day_of_week, month, year, is_holiday, beds_occupied
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} real occupancy records from {csv_path}")
    return df


def load_hospitals(csv_path: str = HOSPITAL_CSV) -> pd.DataFrame:
    """Load hospital metadata CSV (output of Phase 1)."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} hospitals from {csv_path}")
    return df


def generate_training_data(hospital_df: pd.DataFrame,
                           occupancy_df: pd.DataFrame,
                           n_rows: int = 2000,
                           seed: int = SEED) -> pd.DataFrame:
    """
    Generate an augmented training dataset by combining real occupancy
    patterns with hospital metadata.

    Uses the real bed_occupancy_data.csv distribution to calibrate
    synthetic training rows.

    Features:
      - hour_of_day           (0–23)
      - day_of_week           (0–6, from real data)
      - is_holiday            (0/1, from real data)
      - month                 (1–12, from real data)
      - current_occupancy_pct (calibrated from real beds_occupied)
      - incoming_emergencies  (0–10)
      - trauma_center         (0/1)
      - total_beds            (from hospital metadata)

    Target:
      - predicted_available_beds (int)

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.RandomState(seed)

    # Use real occupancy stats for calibration
    real_mean_occ = occupancy_df["beds_occupied"].mean()
    real_std_occ = occupancy_df["beds_occupied"].std()
    real_days = occupancy_df["day_of_week"].values
    real_holidays = occupancy_df["is_holiday"].values

    rows = []
    for _ in range(n_rows):
        # Sample a hospital
        h = hospital_df.iloc[rng.randint(0, len(hospital_df))]
        total_beds = int(h["total_beds"])

        # Time features — mix real patterns with variation
        hour = rng.randint(0, 24)
        day_idx = rng.randint(0, len(real_days))
        day = int(real_days[day_idx])
        is_holiday = int(real_holidays[day_idx])
        month = rng.randint(1, 13)

        # Occupancy — calibrated from real data
        base_occ = rng.normal(real_mean_occ / 300, real_std_occ / 300)
        base_occ = np.clip(base_occ, 0.30, 0.95)

        # Time-of-day effect
        if 7 <= hour <= 10:      # morning rush
            base_occ += 0.05
        elif 16 <= hour <= 20:   # evening surge
            base_occ += 0.07
        elif 0 <= hour <= 5:     # overnight lull
            base_occ -= 0.08

        # Holiday / weekend effect
        if is_holiday:
            base_occ += 0.04
        if day >= 5:
            base_occ -= 0.03

        base_occ = np.clip(base_occ, 0.30, 0.95)

        incoming = rng.randint(0, 11)
        trauma = int(h["trauma_center"])

        # Target: available beds
        occupied = int(total_beds * base_occ)
        emergency_impact = incoming * rng.randint(1, 4)
        available = max(0, total_beds - occupied - emergency_impact)

        rows.append({
            "hospital_id": h["hospital_id"],
            "hour_of_day": hour,
            "day_of_week": day,
            "is_holiday": is_holiday,
            "month": month,
            "current_occupancy_pct": round(base_occ, 3),
            "incoming_emergencies": incoming,
            "trauma_center": trauma,
            "total_beds": total_beds,
            "predicted_available_beds": available,
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Generated {len(df)} training rows "
          f"(calibrated from {len(occupancy_df)} real occupancy records)")
    return df


def train_forecaster(train_df: pd.DataFrame):
    """
    Train a Random Forest Regressor.

    Returns
    -------
    model, X_test, y_test
    """
    feature_cols = [
        "hour_of_day", "day_of_week", "is_holiday", "month",
        "current_occupancy_pct", "incoming_emergencies",
        "trauma_center", "total_beds",
    ]
    X = train_df[feature_cols].values
    y = train_df["predicted_available_beds"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest Regressor trained.")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n===== Evaluation Metrics =====")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    return model, X_test, y_test


def save_model(model, path: str = MODEL_PATH):
    """Save the trained model."""
    joblib.dump(model, path)
    print(f"[✓] Model saved → {path}")


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

_forecast_model = None
_hospital_df = None


def _ensure_loaded():
    """Lazy-load model and hospital data."""
    global _forecast_model, _hospital_df
    if _forecast_model is None:
        _forecast_model = joblib.load(MODEL_PATH)
    if _hospital_df is None:
        _hospital_df = pd.read_csv(HOSPITAL_CSV)


def forecast_hospital_load(hospital_id: str, hour: int,
                           day: int) -> dict:
    """
    Predict hospital bed availability.

    Parameters
    ----------
    hospital_id : str  (e.g. "H001")
    hour : int         (0–23)
    day : int          (0=Mon … 6=Sun)

    Returns
    -------
    dict : {"hospital_id", "predicted_available_beds", "load_status"}
    """
    _ensure_loaded()

    h = _hospital_df[_hospital_df["hospital_id"] == hospital_id]
    if len(h) == 0:
        return {"hospital_id": hospital_id,
                "predicted_available_beds": 0, "load_status": "high"}
    h = h.iloc[0]

    is_weekend = int(day >= 5)
    occ_pct = h["current_occupancy"] / h["total_beds"]
    features = np.array([[
        hour, day, is_weekend, 1,  # month=1 as default
        occ_pct, 3,  # avg incoming
        int(h["trauma_center"]), int(h["total_beds"]),
    ]])

    predicted = int(max(0, _forecast_model.predict(features)[0]))
    ratio = predicted / int(h["total_beds"]) if h["total_beds"] > 0 else 0

    status = "low" if ratio > 0.4 else ("medium" if ratio > 0.2 else "high")

    return {
        "hospital_id": hospital_id,
        "predicted_available_beds": predicted,
        "load_status": status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 5] Hospital Load Forecasting (REAL DATA)")
    print("=" * 55)

    hospital_df = load_hospitals()
    occupancy_df = load_real_occupancy()

    print(f"\n  Real occupancy stats:")
    print(f"    Mean beds occupied : {occupancy_df['beds_occupied'].mean():.0f}")
    print(f"    Min                : {occupancy_df['beds_occupied'].min()}")
    print(f"    Max                : {occupancy_df['beds_occupied'].max()}")

    train_df = generate_training_data(hospital_df, occupancy_df)
    model, X_test, y_test = train_forecaster(train_df)
    save_model(model)

    print("\n--- Sample Forecasts ---")
    for hid, hour, day in [("H001", 8, 0), ("H005", 14, 5), ("H010", 22, 3)]:
        result = forecast_hospital_load(hid, hour, day)
        print(f"  {result}")

    print("\nHospital forecaster ready.")
