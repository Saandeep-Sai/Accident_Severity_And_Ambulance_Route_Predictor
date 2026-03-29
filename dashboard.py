"""
dashboard.py — Phase 8: Streamlit Dashboard
==============================================
Interactive dashboard for visualising the Proactive AI
Emergency Response pipeline output.

Launch: streamlit run dashboard.py

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emergency Response Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark theme overrides */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stMetric label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.8rem !important;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 20px 0 10px 0;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_telemetry():
    """Load telemetry data."""
    if os.path.exists("telemetry_data.csv"):
        return pd.read_csv("telemetry_data.csv")
    return pd.DataFrame()


@st.cache_data
def load_incidents():
    """Load incident log."""
    if os.path.exists("incident_log.csv"):
        df = pd.read_csv("incident_log.csv")
        return df
    return pd.DataFrame()


@st.cache_data
def load_hospitals():
    """Load hospital data."""
    if os.path.exists("hospital_data.csv"):
        return pd.read_csv("hospital_data.csv")
    return pd.DataFrame()


def load_image_safe(path: str):
    """Load an image file if it exists, else return None."""
    if os.path.exists(path):
        return path
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🚨 Emergency Response")
st.sidebar.markdown("**Proactive AI Framework**")
st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("Filters")

severity_filter = st.sidebar.selectbox(
    "Severity",
    ["All", "minor", "moderate", "severe"],
    index=0,
)

vehicle_search = st.sidebar.text_input("Vehicle ID", placeholder="e.g. VH042")

st.sidebar.markdown("---")

# Simulation button
if st.sidebar.button("🔄 Run Simulation (20 rows)", use_container_width=True):
    with st.spinner("Running pipeline simulation …"):
        try:
            import pipeline_controller
            # Clear caches so new data shows up
            st.cache_data.clear()
            if os.path.exists("incident_log.csv"):
                os.remove("incident_log.csv")
            summary = pipeline_controller.run_simulation(20)
            st.sidebar.success(
                f"✅ Done! {summary['anomalies_detected']} anomalies, "
                f"{summary['alerts_sent']} alerts"
            )
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built by Anupama, Saipriya & Vishishta\n\n"
    "Guide: P. Raj Kumar"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
telemetry_df = load_telemetry()
incident_df = load_incidents()
hospital_df = load_hospitals()

# Apply filters
if not incident_df.empty:
    filtered_incidents = incident_df.copy()
    if severity_filter != "All":
        filtered_incidents = filtered_incidents[
            filtered_incidents["severity"] == severity_filter
        ]
    if vehicle_search:
        filtered_incidents = filtered_incidents[
            filtered_incidents["vehicle_id"].str.contains(
                vehicle_search, case=False, na=False
            )
        ]
else:
    filtered_incidents = pd.DataFrame()

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("🚨 Proactive AI — Emergency Response Dashboard")
st.markdown(
    "*Real-time monitoring of road anomalies, incident severity, "
    "and emergency dispatch*"
)

# ---------------------------------------------------------------------------
# Section A — KPI Cards
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">📊 Key Performance Indicators</p>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

if not incident_df.empty:
    total_incidents = len(incident_df[incident_df["is_anomaly"] == True])
    avg_score = incident_df["anomaly_score"].mean()
    avg_eta = incident_df.loc[
        incident_df["is_anomaly"] == True, "eta_min"
    ].mean()
    alerts_ok = len(incident_df[incident_df["alert_sent"] == True])
else:
    total_incidents = 0
    avg_score = 0
    avg_eta = 0
    alerts_ok = 0

with col1:
    st.metric("🔴 Total Incidents", f"{total_incidents}")
with col2:
    st.metric("📈 Avg Anomaly Score", f"{avg_score:.3f}")
with col3:
    st.metric("🚑 Avg ETA (min)", f"{avg_eta:.1f}")
with col4:
    st.metric("📨 Alerts Sent", f"{alerts_ok}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section B — Anomaly Trends
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">📈 Anomaly Trends</p>',
            unsafe_allow_html=True)

bcol1, bcol2 = st.columns(2)

with bcol1:
    st.subheader("Anomaly Distribution in Telemetry")
    if not telemetry_df.empty and "is_anomaly" in telemetry_df.columns:
        if "timestamp" in telemetry_df.columns:
            tdf = telemetry_df.copy()
            tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
            tdf = tdf.dropna(subset=["timestamp"])
            tdf["hour"] = tdf["timestamp"].dt.hour
            hourly = tdf.groupby("hour")["is_anomaly"].sum().reset_index()
            hourly.columns = ["Hour", "Anomalies"]
            st.line_chart(hourly.set_index("Hour"), height=300)
        else:
            st.info("No timestamp column available for time-series chart.")
    else:
        st.info("Run Phase 1 & 7 first to generate telemetry data.")

with bcol2:
    st.subheader("Anomaly Type Breakdown")
    if not incident_df.empty and "top_features" in incident_df.columns:
        # Extract top feature from each incident
        anomaly_incidents = incident_df[incident_df["is_anomaly"] == True].copy()
        if not anomaly_incidents.empty:
            top_feat = anomaly_incidents["top_features"].apply(
                lambda x: str(x).split("|")[0] if pd.notna(x) else "unknown"
            )
            feat_counts = top_feat.value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71",
                       "#9b59b6", "#1abc9c", "#e67e22"]
            ax.pie(feat_counts.values, labels=feat_counts.index,
                   autopct="%1.1f%%", colors=colors[:len(feat_counts)],
                   startangle=140, textprops={"fontsize": 9})
            ax.set_title("Top Anomaly Features", fontsize=12, fontweight="bold")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No anomaly incidents found.")
    else:
        st.info("Run the simulation to generate incident data.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section C — Road Network & Dispatch
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">🗺️ Road Network & Dispatch</p>',
            unsafe_allow_html=True)

ccol1, ccol2 = st.columns([1, 1])

with ccol1:
    st.subheader("City Road Network")
    road_img = load_image_safe("road_network_map.png")
    if road_img:
        st.image(road_img, use_column_width=True)
    else:
        st.info("Run Phase 4 to generate the road network map.")

with ccol2:
    st.subheader("Recent Dispatches")
    if not filtered_incidents.empty:
        anomaly_dispatches = filtered_incidents[
            filtered_incidents["is_anomaly"] == True
        ]
        if not anomaly_dispatches.empty:
            display_cols = ["vehicle_id", "severity", "assigned_hospital",
                           "assigned_ambulance", "eta_min", "timestamp"]
            available_cols = [c for c in display_cols
                            if c in anomaly_dispatches.columns]
            st.dataframe(
                anomaly_dispatches[available_cols].tail(15),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No anomalous dispatches in filtered data.")
    else:
        st.info("No incident data available. Run a simulation first.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section D — Model Explainability
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">🔍 Model Explainability (SHAP)</p>',
            unsafe_allow_html=True)

dcol1, dcol2, dcol3 = st.columns([1, 1, 1])

with dcol1:
    st.subheader("SHAP Feature Importance")
    shap_img = load_image_safe("shap_summary.png")
    if shap_img:
        st.image(shap_img, use_column_width=True)
    else:
        st.info("Run Phase 2 to generate SHAP summary.")

with dcol2:
    st.subheader("ROC Curves (IF vs RF)")
    roc_img = load_image_safe("roc_curves.png")
    if roc_img:
        st.image(roc_img, use_column_width=True)
    else:
        st.info("Run Phase 2 to generate ROC curves.")

with dcol3:
    st.subheader("Training History (CNN)")
    hist_img = load_image_safe("training_history.png")
    if hist_img:
        st.image(hist_img, use_column_width=True)
    else:
        st.info("Run Phase 3 to generate training history.")

# Expandable raw SHAP values for last incident
if not incident_df.empty and "top_features" in incident_df.columns:
    with st.expander("📋 Raw feature ranking — last incident"):
        last = incident_df.iloc[-1]
        st.json({
            "vehicle_id": str(last.get("vehicle_id", "N/A")),
            "anomaly_score": float(last.get("anomaly_score", 0)),
            "severity": str(last.get("severity", "N/A")),
            "top_features": str(last.get("top_features", "")).split("|"),
        })

# ---------------------------------------------------------------------------
# Section D2 — Severity Prediction (Image Upload)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<p class="section-header">📸 Severity Prediction — Upload & Test</p>',
            unsafe_allow_html=True)
st.markdown(
    "Upload an accident image **or** pick one from the validation set to run "
    "the CNN severity classifier in real time."
)

upcol1, upcol2 = st.columns([1, 1])

with upcol1:
    # --- Option 1: Upload your own image ---
    uploaded_file = st.file_uploader(
        "Upload an accident image",
        type=["jpg", "jpeg", "png"],
        key="severity_uploader",
    )

    # --- Option 2: Pick from validation set ---
    st.markdown("**— or pick from validation set —**")
    val_dir = os.path.join("data", "images", "validation")
    sample_options = {"(none)": None}
    for sev in ["minor", "moderate", "severe"]:
        sev_dir = os.path.join(val_dir, sev)
        if os.path.isdir(sev_dir):
            files = sorted([f for f in os.listdir(sev_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in files[:5]:  # show first 5 per class
                label = f"{sev}/{f}"
                sample_options[label] = os.path.join(sev_dir, f)

    sample_choice = st.selectbox(
        "Pick a sample image",
        list(sample_options.keys()),
        index=0,
        key="sample_picker",
    )

with upcol2:
    # --- Run prediction ---
    prediction_result = None
    display_image = None

    if uploaded_file is not None:
        # User uploaded an image
        display_image = uploaded_file
        image_bytes = uploaded_file.getvalue()

        with st.spinner("Running CNN prediction …"):
            try:
                import severity_classifier
                prediction_result = severity_classifier.predict_severity_from_bytes(image_bytes)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif sample_choice != "(none)" and sample_options[sample_choice] is not None:
        # User picked a validation image
        img_path = sample_options[sample_choice]
        display_image = img_path

        with st.spinner("Running CNN prediction …"):
            try:
                import severity_classifier
                prediction_result = severity_classifier.predict_severity(img_path)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if display_image is not None:
        st.image(display_image, caption="Input Image", width=300)

    if prediction_result is not None and prediction_result.get("severity") != "unknown":
        severity = prediction_result["severity"]
        confidence = prediction_result["confidence"]
        probs = prediction_result.get("probabilities", {})

        # Severity badge with color
        severity_colors = {
            "minor": "#27ae60",
            "moderate": "#f39c12",
            "severe": "#e74c3c",
        }
        badge_color = severity_colors.get(severity, "#888")

        st.markdown(
            f'<div style="text-align:center; margin: 15px 0;">'
            f'<span style="background:{badge_color}; color:white; '
            f'padding:10px 30px; border-radius:25px; font-size:1.4rem; '
            f'font-weight:700; letter-spacing:1px;">'
            f'{severity.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align:center; font-size:1.1rem; color:#ccc;">'
            f'Confidence: <strong>{confidence*100:.1f}%</strong></p>',
            unsafe_allow_html=True,
        )

        # Probability bar chart
        if probs:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            classes = list(probs.keys())
            values = [probs[c] * 100 for c in classes]
            colors = [severity_colors.get(c, "#888") for c in classes]
            bars = ax.barh(classes, values, color=colors, edgecolor="white",
                          height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif display_image is None:
        st.info("👆 Upload an image or pick a sample to see the CNN prediction.")

# ---------------------------------------------------------------------------
# Section E — Hospital Overview
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<p class="section-header">🏥 Hospital Overview</p>',
            unsafe_allow_html=True)

if not hospital_df.empty:
    ecol1, ecol2 = st.columns([2, 1])
    with ecol1:
        st.dataframe(hospital_df, use_container_width=True, hide_index=True)
    with ecol2:
        st.subheader("Bed Availability")
        hospital_df_disp = hospital_df.copy()
        hospital_df_disp["available_beds"] = (
            hospital_df_disp["total_beds"] - hospital_df_disp["current_occupancy"]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        top10 = hospital_df_disp.nlargest(10, "available_beds")
        bars = ax.barh(
            top10["hospital_name"].str[:20],
            top10["available_beds"],
            color="#27ae60",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Available Beds")
        ax.set_title("Top 10 Hospitals by Availability", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
else:
    st.info("No hospital data available.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Proactive AI Framework for Early Anomaly Assessment & Rapid Emergency Response<br>"
    "© 2026 — Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)"
    "</div>",
    unsafe_allow_html=True,
)
