import os
import json
import re
import threading
import uuid
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify

from google import genai
from google.genai import types


# ----------------------------
# Config
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PDF_MB = float(os.getenv("MAX_PDF_MB", "15"))

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
app = Flask(__name__)

# ----------------------------
# In-memory job store
# (OK for demo / free Render)
# ----------------------------
jobs: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Helpers
# ----------------------------
def extract_json(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"({[\s\S]*})", text)
    if brace:
        return json.loads(brace.group(1))
    raise ValueError("No JSON found in Gemini output")


def normalize(d: Dict[str, Any]) -> Dict[str, Any]:
    def fnum(v):
        try:
            return float(re.sub(r"[^\d.\-]", "", str(v)))
        except Exception:
            return 0.0

    def fstr(v):
        return str(v).strip() if v else ""

    def flist(v):
        return v if isinstance(v, list) else []

    return {
        "invoice_number": fstr(d.get("invoice_number")),
        "vendor": fstr(d.get("vendor")),
        "invoice_date": fstr(d.get("invoice_date")),
        "due_date": fstr(d.get("due_date")),
        "currency": fstr(d.get("currency", "USD")),
        "subtotal": fnum(d.get("subtotal")),
        "tax": fnum(d.get("tax")),
        "total": fnum(d.get("total")),
        "confidence": min(1.0, max(0.0, fnum(d.get("confidence")))),
        "flags": flist(d.get("flags")),
        "summary": fstr(d.get("summary")),
    }


def run_gemini(job_id: str, pdf_bytes: bytes, context: Dict[str, str]):
    try:
        prompt = f"""
Extract invoice fields and return STRICT JSON with keys:
invoice_number, vendor, invoice_date, due_date,
currency, subtotal, tax, total, confidence, flags, summary.

Email context (may help):
From: {context.get("email_from")}
Subject: {context.get("email_subject")}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction="Return only JSON. No markdown."
            ),
        )

        data = extract_json(response.text)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = normalize(data)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME})


@app.post("/submit_invoice")
def submit_invoice():
    if "invoice_pdf" not in request.files:
        return jsonify({"error": "invoice_pdf missing"}), 400

    pdf = request.files["invoice_pdf"].read()
    if len(pdf) == 0:
        return jsonify({"error": "empty pdf"}), 400

    if len(pdf) / (1024 * 1024) > MAX_PDF_MB:
        return jsonify({"error": "pdf too large"}), 413

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing",
        "result": None,
    }

    context = {
        "email_from": request.form.get("email_from", ""),
        "email_subject": request.form.get("email_subject", ""),
    }

    t = threading.Thread(
        target=run_gemini,
        args=(job_id, pdf, context),
        daemon=True,
    )
    t.start()

    return jsonify({
        "status": "accepted",
        "job_id": job_id
    })


@app.get("/job_status")
def job_status():
    job_id = request.args.get("job_id")
    if not job_id or job_id not in jobs:
        return jsonify({"error": "job_id not found"}), 404

    return jsonify(jobs[job_id])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
