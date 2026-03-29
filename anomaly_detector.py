"""
anomaly_detector.py — Phase 2: Anomaly Detection Module
=========================================================
Dual-model anomaly detection using REAL vehicle telemetry:
  1. Isolation Forest  (unsupervised — trained on normal data only)
  2. Random Forest Classifier (supervised — for high-accuracy classification)

Both models are evaluated; the supervised RF is used for pipeline
inference while Isolation Forest provides unsupervised anomaly scores.
SHAP explains the Random Forest decisions.

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20

# Features from the real telemetry data
FEATURE_COLS = [
    "speed_kmph",
    "engine_temp_c",
    "engine_load_pct",
    "intake_air_temp_c",
    "intake_manifold_pressure",
    "engine_rpm",
    "gps_speed_kmph",
]


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_telemetry(csv_path: str = "telemetry_data.csv") -> pd.DataFrame:
    """Load the preprocessed telemetry CSV."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows from {csv_path}")
    print(f"[INFO] Anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")
    return df


def preprocess(df: pd.DataFrame):
    """
    Preprocess telemetry data for BOTH models.

    Returns
    -------
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote,
    X_train_normal, scaler
    """
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    print(f"[INFO] Clean rows: {len(df_clean):,}")

    X = df_clean[FEATURE_COLS].values
    y = df_clean["is_anomaly"].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # For Isolation Forest: train on NORMAL data only (proper anomaly detection)
    X_train_normal = X_train_sc[y_train == 0]
    print(f"[INFO] Isolation Forest training set: {len(X_train_normal):,} normal samples")

    # For Random Forest: SMOTE-balanced data
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
    print(f"[INFO] Random Forest (SMOTE): {len(X_train_smote):,} samples "
          f"(class 0: {(y_train_smote==0).sum():,}, "
          f"class 1: {(y_train_smote==1).sum():,})")

    return (X_train_sc, X_test_sc, y_train, y_test,
            X_train_smote, y_train_smote, X_train_normal, scaler)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_isolation_forest(X_normal: np.ndarray):
    """
    Train Isolation Forest on NORMAL data only.
    This is the proper approach — the model learns what 'normal'
    looks like and flags deviations as anomalies.
    """
    # Contamination set low since we train on normal data only
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,    # expect very few anomalies in "normal" set
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_normal)
    print("[INFO] Isolation Forest trained (on normal data only).")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train supervised Random Forest Classifier on SMOTE-balanced data.
    This provides high-accuracy anomaly classification.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest Classifier trained.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_isolation_forest(model, X_test, y_test):
    """Evaluate Isolation Forest."""
    raw = model.predict(X_test)
    preds = (raw == -1).astype(int)
    scores = model.decision_function(X_test)

    print("\n" + "=" * 55)
    print("  ISOLATION FOREST (Unsupervised)")
    print("=" * 55)
    print(classification_report(y_test, preds,
                                target_names=["Normal", "Anomaly"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    try:
        auc = roc_auc_score(y_test, -scores)
        print(f"AUC-ROC: {auc:.4f}")
        return auc
    except:
        return 0.0


def evaluate_random_forest(model, X_test, y_test):
    """Evaluate Random Forest Classifier."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("  RANDOM FOREST CLASSIFIER (Supervised)")
    print("=" * 55)
    print(classification_report(y_test, preds,
                                target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)

    try:
        auc = roc_auc_score(y_test, proba)
        print(f"AUC-ROC: {auc:.4f}")
    except:
        auc = 0.0

    return auc, preds, proba


def plot_roc_curves(y_test, if_scores, rf_proba, save_path="roc_curves.png"):
    """Plot ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Isolation Forest ROC
    fpr_if, tpr_if, _ = roc_curve(y_test, -if_scores)
    auc_if = roc_auc_score(y_test, -if_scores)
    ax.plot(fpr_if, tpr_if, 'b-', linewidth=2,
            label=f'Isolation Forest (AUC = {auc_if:.3f})')

    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_rf = roc_auc_score(y_test, rf_proba)
    ax.plot(fpr_rf, tpr_rf, 'r-', linewidth=2,
            label=f'Random Forest (AUC = {auc_rf:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.500)')

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Anomaly Detection Models", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] ROC curves saved → {save_path}")


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def generate_shap_summary(model, X_test, save_path="shap_summary.png"):
    """Generate SHAP summary for the Random Forest Classifier."""
    print("[INFO] Computing SHAP values …")
    n_shap = min(500, len(X_test))
    X_shap = X_test[:n_shap]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # For binary classifier, shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_shap, feature_names=FEATURE_COLS,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Random Forest Classifier)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] SHAP summary saved → {save_path}")


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(if_model, rf_model, scaler):
    """Save both models and scaler."""
    joblib.dump(if_model, "isolation_forest.pkl")
    joblib.dump(rf_model, "random_forest_clf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("[✓] isolation_forest.pkl saved")
    print("[✓] random_forest_clf.pkl saved")
    print("[✓] scaler.pkl saved")


# ---------------------------------------------------------------------------
# Single-row inference API
# ---------------------------------------------------------------------------

_if_model = None
_rf_model = None
_scaler = None


def _ensure_loaded():
    """Lazy-load models and scaler."""
    global _if_model, _rf_model, _scaler
    if _scaler is None:
        _scaler = joblib.load("scaler.pkl")
    if _rf_model is None:
        _rf_model = joblib.load("random_forest_clf.pkl")
    if _if_model is None:
        _if_model = joblib.load("isolation_forest.pkl")


def detect_anomaly(row: dict) -> dict:
    """
    Run anomaly detection on a single telemetry row.

    Uses the supervised Random Forest for classification and
    Isolation Forest for anomaly scoring.

    Parameters
    ----------
    row : dict
        Must contain keys matching FEATURE_COLS.

    Returns
    -------
    dict
        {
            "is_anomaly": bool,
            "anomaly_score": float,
            "confidence": float,
            "top_features": list[str]
        }
    """
    _ensure_loaded()

    features = np.array([[row.get(c, 0) for c in FEATURE_COLS]])
    scaled = _scaler.transform(features)

    # Supervised prediction (main decision)
    rf_pred = _rf_model.predict(scaled)[0]
    rf_proba = _rf_model.predict_proba(scaled)[0]
    confidence = float(max(rf_proba))

    # Unsupervised anomaly score
    if_score = _if_model.decision_function(scaled)[0]

    # Feature importance from RF
    importances = _rf_model.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances),
                    key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in ranked]

    return {
        "is_anomaly": bool(rf_pred == 1),
        "anomaly_score": float(round(-if_score, 4)),
        "confidence": float(round(confidence, 4)),
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 2] Anomaly Detection Module (REAL DATA)")
    print("=" * 55)

    df = load_telemetry()

    (X_train, X_test, y_train, y_test,
     X_train_smote, y_train_smote, X_train_normal, scaler) = preprocess(df)

    # Train both models
    if_model = train_isolation_forest(X_train_normal)
    rf_model = train_random_forest(X_train_smote, y_train_smote)

    # Evaluate both
    if_auc = evaluate_isolation_forest(if_model, X_test, y_test)
    rf_auc, rf_preds, rf_proba = evaluate_random_forest(rf_model, X_test, y_test)

    # ROC curve comparison
    if_scores = if_model.decision_function(X_test)
    plot_roc_curves(y_test, if_scores, rf_proba)

    # SHAP on the Random Forest (the model we actually use)
    generate_shap_summary(rf_model, X_test)

    # Save
    save_artifacts(if_model, rf_model, scaler)

    # Smoke test
    print("\n--- Smoke Test ---")
    sample = {
        "speed_kmph": 115.0, "engine_temp_c": 95.0,
        "engine_load_pct": 92.0, "intake_air_temp_c": 55.0,
        "intake_manifold_pressure": 200.0, "engine_rpm": 4500.0,
        "gps_speed_kmph": 110.0,
    }
    result = detect_anomaly(sample)
    print(f"  detect_anomaly(high-risk row) → {result}")

    sample_normal = {
        "speed_kmph": 25.0, "engine_temp_c": 70.0,
        "engine_load_pct": 30.0, "intake_air_temp_c": 35.0,
        "intake_manifold_pressure": 100.0, "engine_rpm": 900.0,
        "gps_speed_kmph": 24.0,
    }
    result2 = detect_anomaly(sample_normal)
    print(f"  detect_anomaly(normal row)    → {result2}")

    print("\nAnomaly detector ready.")
