"""
app.py — Hugging Face Spaces entry point.
Wraps the main dashboard for HF Spaces compatibility.
"""
import subprocess
import sys

# HF Spaces runs this file directly — Streamlit expects to be
# launched via 'streamlit run', so we do that here.
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])
