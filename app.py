#!/usr/bin/env python3

"""
kanboard_huggingface_agent.py
Fetch tasks for one Kanboard project, classify them, call a Hugging Face LLM for summary,
and send the report by email.
"""

import os
import sys
import logging
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from base64 import b64encode

# Hugging Face client
from huggingface_hub import InferenceClient

# Load .env
load_dotenv()

# ---- Config from env ----
HF_API_KEY = os.getenv("HF_API_KEY")
KANBOARD_URL = os.getenv("KB_URL")
KANBOARD_USER = os.getenv("KB_USER")
KANBOARD_TOKEN = os.getenv("KB_TOKEN")
PROJECT_ID = int(os.getenv("KB_PROJECT_ID", "16"))

EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "ttc-prod.smtps.jp")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")  # comma-separated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kanboard-hf-agent")

if not all([HF_API_KEY, KANBOARD_URL, KANBOARD_USER, KANBOARD_TOKEN, PROJECT_ID, EMAIL_USER, EMAIL_PASS, EMAIL_TO]):
    logger.error("Missing required environment variables. Exiting.")
    sys.exit(2)

# ---- Hugging Face Client ----
hf_client = InferenceClient(token=HF_API_KEY)

# ---- Kanboard helper ----


def kb_headers():
    auth = f"{KANBOARD_USER}:{KANBOARD_TOKEN}"
    token = b64encode(auth.encode()).decode()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {token}"
    }


def kb_call(method, params=None):
    payload = {"jsonrpc": "2.0", "method": method, "id": 1}
    if params:
        payload["params"] = params
    r = requests.post(KANBOARD_URL, json=payload,
                      headers=kb_headers(), timeout=30)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"Kanboard error: {j['error']}")
    return j.get("result")


def fetch_tasks(project_id):
    tasks = kb_call("getAllTasks", {"project_id": project_id})
    for t in tasks:
        if not t.get("column_name"):
            task_info = kb_call("getTask", {"task_id": t["id"]})
            t["column_name"] = task_info.get("column_title") or "Unknown"
    return tasks

# ---- Task Classification ----


def classify_tasks(tasks):
    wip_columns = ["work in progress", "dev",
                   "qc", "uat", "staging", "production"]
    blocked_columns = []
    wip, blocked = [], []

    for t in tasks:
        col = (t.get("column_name") or "").strip().lower()
        if col in [c.lower() for c in wip_columns]:
            wip.append(t)
        elif col in [c.lower() for c in blocked_columns]:
            blocked.append(t)

    return wip, blocked

# ---- Build report ----


def build_report_text(project_id, wip, blocked):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Kanboard AI Report — Project {project_id}",
        f"Generated: {now}", ""
    ]
    lines.append(
        f"Summary counts: InProgress={len(wip)}, Blocked={len(blocked)}\n")

    def section(title, items):
        lines.append(title + ":")
        if not items:
            lines.append("  None\n")
            return
        for t in items:
            due = t.get("date_due") and datetime.fromtimestamp(
                int(t["date_due"])).strftime("%Y-%m-%d") or "No due"
            lines.append(
                f"  - {t.get('title')} (id:{t.get('id')} | due:{due} | column:{t.get('column_name')})"
            )
        lines.append("")

    section("Work In Progress", wip)
    section("Blocked / On Hold", blocked)
    return "\n".join(lines)

# ---- Hugging Face Summarization ----


def hf_summary(report_text):
    prompt = f"""
You are a project management assistant.

Summarize the following Kanboard report in:
1. 3 key risk points
2. 3 recommended actions
3. 2 productivity improvement tips

Report:
{report_text}
"""

    try:
        response = hf_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.4
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        return f"Unable to generate summary. Error: {str(e)}"

# ---- Save report locally ----


def save_report(report_text, summary_text=None, folder="reports"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"kanboard_report_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        if summary_text:
            f.write("===== HF Summary =====\n")
            f.write(summary_text + "\n\n")
        f.write("===== Kanboard Raw Report =====\n")
        f.write(report_text)
    logger.info(f"Report saved locally: {filename}")
    return filename

# ---- Email ----


def send_email(subject, body_plain):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body_plain, "plain"))

    s = smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30)
    s.ehlo()
    if EMAIL_SMTP_PORT == 587:
        s.starttls()
    s.login(EMAIL_USER, EMAIL_PASS)
    s.sendmail(EMAIL_USER, [x.strip()
               for x in EMAIL_TO.split(",")], msg.as_string())
    s.quit()
    logger.info("Email sent successfully")

# ---- Runner ----


def run(test_only=True):
    tasks = fetch_tasks(PROJECT_ID)
    wip, blocked = classify_tasks(tasks)
    plain = build_report_text(PROJECT_ID, wip, blocked)

    try:
        summary = hf_summary(plain)
    except Exception as e:
        logger.exception("HF summarization failed")
        summary = None

    save_report(plain, summary)

    if test_only:
        print("===== Kanboard Raw Report =====")
        print(plain)
        if summary:
            print("\n===== HF Summary =====")
            print(summary)
        return

    body = f"Hugging Face summary:\n\n{summary}\n\n---\n\n{plain}" if summary else plain
    subject = f"Kanboard Report - Project {PROJECT_ID}"
    send_email(subject, body)
    logger.info("Report sent: WIP=%d Blocked=%d", len(wip), len(blocked))


if __name__ == "__main__":
    run(test_only=False)
