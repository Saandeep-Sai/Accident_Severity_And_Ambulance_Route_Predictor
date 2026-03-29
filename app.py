"""
app.py — Phase 8: Gradio Dashboard (Hugging Face Spaces)
==========================================================
Interactive dashboard for visualising the Proactive AI
Emergency Response pipeline output.

Launch:  python app.py
HF URL:  https://huggingface.co/spaces/<your-space>

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
}
.kpi-icon { font-size: 2rem; margin-bottom: 6px; }
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.85rem;
    color: #a0aec0;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section Headers */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 10px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(102, 126, 234, 0.2);
}

/* Severity Badges */
.badge {
    display: inline-block;
    padding: 10px 32px;
    border-radius: 25px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: white;
    text-transform: uppercase;
}
.badge-minor    { background: linear-gradient(135deg, #27ae60, #2ecc71); }
.badge-moderate { background: linear-gradient(135deg, #f39c12, #e67e22); }
.badge-severe   { background: linear-gradient(135deg, #e74c3c, #c0392b); }

/* Footer */
.footer {
    text-align: center;
    color: #666;
    font-size: 0.82rem;
    padding: 20px 0 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 30px;
}
"""

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_csv_safe(path):
    """Load a CSV file if it exists."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_image_safe(path):
    """Return path if file exists, else None."""
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# KPI Helpers
# ---------------------------------------------------------------------------

def build_kpi_html():
    """Build the 4 KPI cards as HTML."""
    incident_df = load_csv_safe("incident_log.csv")

    if not incident_df.empty and "is_anomaly" in incident_df.columns:
        total = int(incident_df["is_anomaly"].sum())
        avg_score = float(incident_df["anomaly_score"].mean())
        anomalies = incident_df[incident_df["is_anomaly"] == True]
        avg_eta = float(anomalies["eta_min"].mean()) if "eta_min" in anomalies.columns and not anomalies.empty else 0
        alerts = int(incident_df["alert_sent"].sum()) if "alert_sent" in incident_df.columns else 0
    else:
        total, avg_score, avg_eta, alerts = 0, 0.0, 0.0, 0

    cards = [
        ("🔴", f"{total}",      "Total Incidents"),
        ("📈", f"{avg_score:.3f}", "Avg Anomaly Score"),
        ("🚑", f"{avg_eta:.1f}",   "Avg ETA (min)"),
        ("📨", f"{alerts}",     "Alerts Sent"),
    ]

    html = '<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:10px 0 20px;">'
    for icon, value, label in cards:
        html += f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>"""
    html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Chart Generators
# ---------------------------------------------------------------------------

def make_anomaly_distribution_plot():
    """Hourly anomaly distribution from telemetry."""
    df = load_csv_safe("telemetry_data.csv")
    if df.empty or "is_anomaly" not in df.columns or "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["is_anomaly"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(hourly["hour"], hourly["is_anomaly"], alpha=0.3, color="#667eea")
    ax.plot(hourly["hour"], hourly["is_anomaly"], color="#667eea", linewidth=2.5, marker="o", markersize=5)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Anomaly Count", fontsize=11)
    ax.set_title("Anomaly Distribution by Hour", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#a0aec0")
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig


def make_anomaly_type_pie():
    """Pie chart of anomaly feature types."""
    df = load_csv_safe("incident_log.csv")
    if df.empty or "top_features" not in df.columns:
        return None

    anomalies = df[df["is_anomaly"] == True].copy()
    if anomalies.empty:
        return None

    top_feat = anomalies["top_features"].apply(
        lambda x: str(x).split("|")[0] if pd.notna(x) else "unknown"
    )
    feat_counts = top_feat.value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#e67e22"]
    wedges, texts, autotexts = ax.pie(
        feat_counts.values, labels=feat_counts.index,
        autopct="%1.1f%%", colors=colors[:len(feat_counts)],
        startangle=140, textprops={"fontsize": 10, "color": "#e2e8f0"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Top Anomaly Features", fontsize=13, fontweight="bold", color="#e2e8f0")
    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout()
    return fig


def make_bed_availability_chart():
    """Horizontal bar chart of hospital bed availability."""
    df = load_csv_safe("hospital_data.csv")
    if df.empty:
        return None

    df["available_beds"] = df["total_beds"] - df["current_occupancy"]
    top10 = df.nlargest(10, "available_beds")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(
        top10["hospital_name"].str[:22], top10["available_beds"],
        color="#27ae60", edgecolor="#1a1a2e", linewidth=0.8,
    )
    for bar, val in zip(bars, top10["available_beds"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9, color="#a0aec0", fontweight="bold")
    ax.set_xlabel("Available Beds", fontsize=11, color="#a0aec0")
    ax.set_title("Top 10 Hospitals by Availability", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.invert_yaxis()
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#a0aec0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Severity Prediction (CNN)
# ---------------------------------------------------------------------------

def predict_severity_from_image(image):
    """
    Gradio callback: accept a PIL/numpy image, run CNN, return label + chart.
    """
    if image is None:
        return (
            '<div style="text-align:center;color:#888;padding:30px;">'
            '👆 Upload an accident image to get a prediction.</div>',
            None,
        )

    import cv2
    import severity_classifier

    # Convert PIL → numpy → BGR for OpenCV
    if hasattr(image, "convert"):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_norm = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    severity_classifier._ensure_model_loaded()
    preds = severity_classifier._severity_model.predict(img_batch, verbose=0)[0]
    classes = severity_classifier.CLASSES
    idx = int(np.argmax(preds))
    severity = classes[idx]
    confidence = float(preds[idx])

    # Build result HTML
    badge_class = f"badge-{severity}"
    result_html = f"""
    <div style="text-align:center; padding:20px;">
        <div class="badge {badge_class}">{severity.upper()}</div>
        <p style="font-size:1.15rem; color:#a0aec0; margin-top:12px;">
            Confidence: <strong style="color:#e2e8f0;">{confidence*100:.1f}%</strong>
        </p>
    </div>
    """

    # Probability bar chart
    probs = {classes[i]: float(preds[i]) * 100 for i in range(len(classes))}
    severity_colors = {"minor": "#27ae60", "moderate": "#f39c12", "severe": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(5, 2.5))
    c_list = list(probs.keys())
    v_list = list(probs.values())
    colors = [severity_colors.get(c, "#888") for c in c_list]
    bars = ax.barh(c_list, v_list, color=colors, edgecolor="#1a1a2e", height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color="#a0aec0")
    ax.set_title("Class Probabilities", fontsize=12, fontweight="bold", color="#e2e8f0")
    for bar, val in zip(bars, v_list):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold", color="#e2e8f0")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#a0aec0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.tight_layout()

    return result_html, fig


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

def run_simulation():
    """Run the pipeline simulation and return updated data."""
    try:
        if os.path.exists("incident_log.csv"):
            os.remove("incident_log.csv")

        import importlib
        import pipeline_controller
        importlib.reload(pipeline_controller)

        summary = pipeline_controller.run_simulation(20)

        msg = (f"✅ Simulation complete! "
               f"{summary['anomalies_detected']} anomalies detected, "
               f"{summary['alerts_sent']} alerts sent.")
        return (
            build_kpi_html(),
            make_anomaly_distribution_plot(),
            make_anomaly_type_pie(),
            get_dispatch_dataframe(),
            msg,
        )
    except Exception as e:
        return (
            build_kpi_html(),
            make_anomaly_distribution_plot(),
            make_anomaly_type_pie(),
            get_dispatch_dataframe(),
            f"❌ Error: {e}",
        )


def get_dispatch_dataframe():
    """Load and format the dispatch/incident dataframe."""
    df = load_csv_safe("incident_log.csv")
    if df.empty:
        return pd.DataFrame({"Info": ["No incident data. Run a simulation."]})

    anomalies = df[df["is_anomaly"] == True].copy()
    if anomalies.empty:
        return pd.DataFrame({"Info": ["No anomalies detected in this run."]})

    display_cols = ["vehicle_id", "severity", "assigned_hospital",
                    "assigned_ambulance", "eta_min"]
    available = [c for c in display_cols if c in anomalies.columns]
    return anomalies[available].tail(15).reset_index(drop=True)


def get_hospital_dataframe():
    """Load hospital data."""
    df = load_csv_safe("hospital_data.csv")
    if df.empty:
        return pd.DataFrame({"Info": ["No hospital data available."]})
    cols = ["hospital_id", "hospital_name", "total_beds",
            "current_occupancy", "trauma_center"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def get_last_incident_json():
    """Return the last incident's SHAP ranking as JSON string."""
    df = load_csv_safe("incident_log.csv")
    if df.empty or "top_features" not in df.columns:
        return "No incident data available."

    last = df.iloc[-1]
    info = {
        "vehicle_id": str(last.get("vehicle_id", "N/A")),
        "anomaly_score": float(last.get("anomaly_score", 0)),
        "severity": str(last.get("severity", "N/A")),
        "top_features": str(last.get("top_features", "")).split("|"),
    }
    return json.dumps(info, indent=2)


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------

def build_app():
    """Construct the full Gradio Blocks dashboard."""

    with gr.Blocks(
        title="🚨 Proactive AI — Emergency Response",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0e1117",
            body_background_fill_dark="#0e1117",
            block_background_fill="#1a1a2e",
            block_background_fill_dark="#1a1a2e",
            block_border_color="rgba(102,126,234,0.2)",
            block_border_color_dark="rgba(102,126,234,0.2)",
            block_label_text_color="#a0aec0",
            block_label_text_color_dark="#a0aec0",
            block_title_text_color="#e2e8f0",
            block_title_text_color_dark="#e2e8f0",
            body_text_color="#e2e8f0",
            body_text_color_dark="#e2e8f0",
            button_primary_background_fill="linear-gradient(135deg, #667eea, #764ba2)",
            button_primary_background_fill_dark="linear-gradient(135deg, #667eea, #764ba2)",
            button_primary_text_color="white",
            input_background_fill="#16213e",
            input_background_fill_dark="#16213e",
        ),
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 5px;">
            <h1 style="font-size:2.2rem; font-weight:800; margin:0;
                background:linear-gradient(135deg,#667eea,#764ba2);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🚨 Proactive AI — Emergency Response Dashboard
            </h1>
            <p style="color:#a0aec0; margin-top:6px; font-size:0.95rem;">
                Real-time monitoring of road anomalies, incident severity, and emergency dispatch
            </p>
        </div>
        """)

        # ── KPIs ──
        kpi_html = gr.HTML(value=build_kpi_html)

        # ── Tabs ──
        with gr.Tabs():

            # ============================================================
            # TAB 1: Dashboard Overview
            # ============================================================
            with gr.Tab("📊 Dashboard", id="dashboard"):
                gr.HTML('<div class="section-title">📈 Anomaly Trends</div>')

                with gr.Row():
                    anomaly_dist_plot = gr.Plot(
                        value=make_anomaly_distribution_plot,
                        label="Anomaly Distribution by Hour",
                    )
                    anomaly_pie_plot = gr.Plot(
                        value=make_anomaly_type_pie,
                        label="Anomaly Type Breakdown",
                    )

                gr.HTML('<div class="section-title">🗺️ Road Network & Dispatch</div>')

                with gr.Row():
                    road_img_path = load_image_safe("road_network_map.png")
                    if road_img_path:
                        gr.Image(
                            value=road_img_path,
                            label="City Road Network",
                            interactive=False,
                        )
                    else:
                        gr.HTML('<p style="color:#888;">Run Phase 4 to generate road network map.</p>')

                    dispatch_df = gr.Dataframe(
                        value=get_dispatch_dataframe,
                        label="Recent Dispatches",
                        interactive=False,
                    )

                # Simulation controls
                with gr.Row():
                    sim_btn = gr.Button(
                        "🔄 Run Simulation (20 rows)",
                        variant="primary",
                        size="lg",
                    )
                    sim_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Click to run pipeline simulation…",
                    )

                sim_btn.click(
                    fn=run_simulation,
                    inputs=[],
                    outputs=[kpi_html, anomaly_dist_plot, anomaly_pie_plot,
                             dispatch_df, sim_status],
                )

            # ============================================================
            # TAB 2: Severity Prediction (⭐ Star Feature)
            # ============================================================
            with gr.Tab("📸 Severity Prediction", id="predict"):
                gr.HTML('<div class="section-title">📸 Upload & Predict Accident Severity</div>')
                gr.Markdown(
                    "Upload an accident scene image and the **MobileNetV2 CNN** "
                    "will classify it as **Minor**, **Moderate**, or **Severe** in real time."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Accident Image",
                            type="pil",
                            height=350,
                        )

                        # Sample images from validation set
                        val_dir = os.path.join("data", "images", "validation")
                        example_images = []
                        for sev in ["minor", "moderate", "severe"]:
                            sev_dir = os.path.join(val_dir, sev)
                            if os.path.isdir(sev_dir):
                                files = sorted([
                                    f for f in os.listdir(sev_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                                ])
                                for f in files[:3]:
                                    example_images.append(
                                        os.path.join(sev_dir, f)
                                    )

                        if example_images:
                            gr.Examples(
                                examples=example_images,
                                inputs=input_image,
                                label="Or try a sample from the validation set",
                            )

                    with gr.Column(scale=1):
                        prediction_html = gr.HTML(
                            value='<div style="text-align:center;color:#666;padding:60px 0;">'
                                  '👆 Upload an image to see the CNN prediction.</div>',
                        )
                        prob_chart = gr.Plot(label="Class Probabilities")

                input_image.change(
                    fn=predict_severity_from_image,
                    inputs=[input_image],
                    outputs=[prediction_html, prob_chart],
                )

            # ============================================================
            # TAB 3: Model Performance
            # ============================================================
            with gr.Tab("🔬 Model Performance", id="models"):
                gr.HTML('<div class="section-title">🔍 Model Explainability & Performance</div>')

                with gr.Row():
                    shap_path = load_image_safe("shap_summary.png")
                    if shap_path:
                        gr.Image(
                            value=shap_path,
                            label="SHAP Feature Importance (Random Forest)",
                            interactive=False,
                        )
                    else:
                        gr.HTML('<p style="color:#888;">Run Phase 2 to generate SHAP summary.</p>')

                    roc_path = load_image_safe("roc_curves.png")
                    if roc_path:
                        gr.Image(
                            value=roc_path,
                            label="ROC Curves — Isolation Forest vs Random Forest",
                            interactive=False,
                        )
                    else:
                        gr.HTML('<p style="color:#888;">Run Phase 2 to generate ROC curves.</p>')

                with gr.Row():
                    hist_path = load_image_safe("training_history.png")
                    if hist_path:
                        gr.Image(
                            value=hist_path,
                            label="CNN Training History (Two-Stage Fine-Tuning)",
                            interactive=False,
                        )
                    else:
                        gr.HTML('<p style="color:#888;">Run Phase 3 to generate training history.</p>')

                with gr.Accordion("📋 Raw Feature Ranking — Last Incident", open=False):
                    gr.Textbox(
                        value=get_last_incident_json,
                        label="SHAP Feature Ranking (JSON)",
                        lines=8,
                        interactive=False,
                    )

            # ============================================================
            # TAB 4: Hospital Overview
            # ============================================================
            with gr.Tab("🏥 Hospitals", id="hospitals"):
                gr.HTML('<div class="section-title">🏥 Hospital Network Overview</div>')

                with gr.Row():
                    hospital_table = gr.Dataframe(
                        value=get_hospital_dataframe,
                        label="Hospital Capacity & Status",
                        interactive=False,
                    )
                    bed_chart = gr.Plot(
                        value=make_bed_availability_chart,
                        label="Bed Availability",
                    )

        # ── Footer ──
        gr.HTML("""
        <div class="footer">
            Proactive AI Framework for Early Anomaly Assessment & Rapid Emergency Response<br>
            © 2026 — Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A) |
            Guide: P. Raj Kumar | MLRITM, Hyderabad
        </div>
        """)

    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
