"""
alert_system.py — Phase 6: Alert System (SMTP Email)
======================================================
Implements an emergency alert system using SMTP email.
Falls back to logging when credentials are unavailable.
Twilio SMS support is deferred — use MOCK_MODE for testing.

Authors : Anupama Boya, Atukula Saipriya, Nalla Vishishta (CSE-A)
Guide   : P. Raj Kumar
"""

import os
import csv
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration from .env
# ---------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Alert log file
ALERT_LOG = "alert_log.csv"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


def _init_alert_log():
    """Create the alert log CSV with headers if it doesn't exist."""
    if not os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "vehicle_id", "severity", "anomaly_type",
                "channel", "status", "message",
            ])


def _log_alert(incident: dict, channel: str, status: str, message: str):
    """Append an entry to the alert log CSV."""
    _init_alert_log()
    with open(ALERT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            incident.get("vehicle_id", "N/A"),
            incident.get("severity", "N/A"),
            incident.get("anomaly_type", "N/A"),
            channel,
            status,
            message,
        ])


def _format_email_html(incident: dict) -> str:
    """
    Format a structured HTML email body for an incident alert.

    Parameters
    ----------
    incident : dict
        Keys: vehicle_id, location, severity, anomaly_type, timestamp

    Returns
    -------
    str
        HTML string.
    """
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #e74c3c, #c0392b);
                    padding: 20px; border-radius: 8px 8px 0 0; color: white;">
            <h2 style="margin: 0;">🚨 Emergency Alert</h2>
            <p style="margin: 5px 0 0;">Proactive AI Framework — Incident Report</p>
        </div>
        <div style="padding: 20px; border: 1px solid #ddd; border-top: none;
                    border-radius: 0 0 8px 8px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold; color: #555;">
                        Vehicle ID</td>
                    <td style="padding: 8px;">{incident.get('vehicle_id', 'N/A')}</td>
                </tr>
                <tr style="background: #f9f9f9;">
                    <td style="padding: 8px; font-weight: bold; color: #555;">
                        Location</td>
                    <td style="padding: 8px;">{incident.get('location', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold; color: #555;">
                        Severity</td>
                    <td style="padding: 8px;">
                        <span style="background: {'#e74c3c' if incident.get('severity') == 'severe'
                            else '#f39c12' if incident.get('severity') == 'moderate'
                            else '#27ae60'};
                            color: white; padding: 3px 10px; border-radius: 12px;">
                            {incident.get('severity', 'N/A').upper()}
                        </span>
                    </td>
                </tr>
                <tr style="background: #f9f9f9;">
                    <td style="padding: 8px; font-weight: bold; color: #555;">
                        Anomaly Type</td>
                    <td style="padding: 8px;">{incident.get('anomaly_type', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold; color: #555;">
                        Timestamp</td>
                    <td style="padding: 8px;">{incident.get('timestamp', 'N/A')}</td>
                </tr>
            </table>
            <hr style="border: none; border-top: 1px solid #eee; margin: 15px 0;">
            <p style="color: #888; font-size: 12px;">
                This alert was generated automatically by the Proactive AI
                Framework for Early Anomaly Assessment and Rapid Emergency Response.
            </p>
        </div>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Email alert
# ---------------------------------------------------------------------------

def send_email_alert(incident: dict) -> bool:
    """
    Send a formatted HTML email alert for an incident.

    Parameters
    ----------
    incident : dict
        Must contain: vehicle_id, location, severity, anomaly_type, timestamp.

    Returns
    -------
    bool
        True if the email was sent successfully, False otherwise.
    """
    if MOCK_MODE:
        print(f"[MOCK EMAIL] Alert for {incident.get('vehicle_id', '?')}: "
              f"Severity={incident.get('severity', '?')}, "
              f"Type={incident.get('anomaly_type', '?')}")
        _log_alert(incident, "email", "mock_success",
                   "Mock mode — email printed to console")
        return True

    if not SMTP_USER or not SMTP_PASSWORD or not ALERT_EMAIL_TO:
        print("[WARN] SMTP credentials not configured. Skipping email.")
        _log_alert(incident, "email", "skipped", "No SMTP credentials")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"🚨 Emergency Alert — {incident.get('severity', 'N/A').upper()} "
                      f"— Vehicle {incident.get('vehicle_id', '?')}")
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL_TO

    html_body = _format_email_html(incident)
    msg.attach(MIMEText(html_body, "html"))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            print(f"[✓] Email alert sent to {ALERT_EMAIL_TO}")
            _log_alert(incident, "email", "success", f"Sent on attempt {attempt}")
            return True
        except Exception as e:
            print(f"[WARN] Email attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    _log_alert(incident, "email", "failed", "All retries exhausted")
    return False


# ---------------------------------------------------------------------------
# SMS alert (placeholder — Twilio deferred)
# ---------------------------------------------------------------------------

def send_sms_alert(incident: dict) -> bool:
    """
    Send an SMS alert for an incident.

    NOTE: Twilio integration is deferred. This function currently
    operates in mock/print mode only.

    Parameters
    ----------
    incident : dict
        Must contain: vehicle_id, location, severity, anomaly_type, timestamp.

    Returns
    -------
    bool
        True if the SMS was sent (or mocked) successfully.
    """
    sms_body = (
        f"🚨 EMERGENCY ALERT\n"
        f"Vehicle: {incident.get('vehicle_id', 'N/A')}\n"
        f"Severity: {incident.get('severity', 'N/A').upper()}\n"
        f"Type: {incident.get('anomaly_type', 'N/A')}\n"
        f"Location: {incident.get('location', 'N/A')}\n"
        f"Time: {incident.get('timestamp', 'N/A')}"
    )

    print(f"[MOCK SMS] {sms_body}")
    _log_alert(incident, "sms", "mock_success",
               "Twilio deferred — SMS printed to console")
    return True


# ---------------------------------------------------------------------------
# Combined trigger
# ---------------------------------------------------------------------------

def trigger_alert(incident: dict) -> dict:
    """
    Trigger an alert for a detected incident.

    Tries email first. SMS is currently in mock mode (Twilio deferred).

    Parameters
    ----------
    incident : dict
        Incident details.

    Returns
    -------
    dict
        {"sms_sent": bool, "email_sent": bool, "timestamp": str}
    """
    ts = datetime.now().isoformat()

    # Try email
    email_ok = send_email_alert(incident)

    # Try SMS (mock for now)
    sms_ok = send_sms_alert(incident)

    return {
        "sms_sent": sms_ok,
        "email_sent": email_ok,
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Phase 6] Alert System")
    print("=" * 50)

    _init_alert_log()

    sample_incident = {
        "vehicle_id": "VH042",
        "location": "Intersection_12, Hyderabad",
        "severity": "severe",
        "anomaly_type": "overspeeding + brake failure",
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n[INFO] MOCK_MODE = {MOCK_MODE}")
    print(f"[INFO] Testing alert with sample incident …\n")

    result = trigger_alert(sample_incident)

    print(f"\n--- Alert Result ---")
    print(f"  SMS sent   : {result['sms_sent']}")
    print(f"  Email sent : {result['email_sent']}")
    print(f"  Timestamp  : {result['timestamp']}")

    print(f"\n[✓] Alert log written to {ALERT_LOG}")
    print("\nAlert system ready.")
