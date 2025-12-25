import os
import json
import re
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify

from google import genai
from google.genai import types


# ----------------------------
# Config
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set this on Render
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # fast & good for docs
MAX_PDF_MB = float(os.getenv("MAX_PDF_MB", "15"))  # safety
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)


# ----------------------------
# Helpers
# ----------------------------
def clamp_str(x: Optional[str], limit: int = 5000) -> str:
    if not x:
        return ""
    x = str(x)
    return x[:limit]


def extract_json(text: str) -> Dict[str, Any]:
    """
    Try hard to extract a JSON object from model output.
    Handles cases where model wraps JSON in ```json ... ``` or adds extra text.
    """
    if not text:
        raise ValueError("Empty model response")

    # 1) strip code fences if present
    fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S)
    if fenced:
        return json.loads(fenced.group(1))

    # 2) find the first {...} block (best-effort)
    brace = re.search(r"({[\s\S]*})", text)
    if brace:
        return json.loads(brace.group(1))

    # 3) give up
    raise ValueError(f"Could not find JSON in model output: {text[:400]}...")


def normalize_invoice_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required keys exist and types are sane.
    """
    def fnum(v, default=0.0):
        try:
            if v is None:
                return default
            if isinstance(v, (int, float)):
                return float(v)
            # remove commas / currency symbols
            s = str(v).strip()
            s = re.sub(r"[^\d.\-]", "", s)
            return float(s) if s else default
        except Exception:
            return default

    def fstr(v, default=""):
        return str(v).strip() if v is not None else default

    def flist(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if str(v).strip() else []

    out = {
        "invoice_number": fstr(d.get("invoice_number")),
        "vendor": fstr(d.get("vendor")),
        "invoice_date": fstr(d.get("invoice_date")),   # keep ISO string if possible
        "due_date": fstr(d.get("due_date")),
        "currency": fstr(d.get("currency", "USD")),
        "subtotal": fnum(d.get("subtotal")),
        "tax": fnum(d.get("tax")),
        "total": fnum(d.get("total")),
        "confidence": max(0.0, min(1.0, fnum(d.get("confidence"), default=0.0))),
        "flags": flist(d.get("flags")),
        "summary": fstr(d.get("summary")),
    }
    return out


def gemini_extract_invoice(pdf_bytes: bytes, context: Dict[str, str]) -> Dict[str, Any]:
    """
    Calls Gemini with the PDF as inline bytes and requests strict JSON output.
    """
    system_instruction = (
        "You are a meticulous accounting assistant. "
        "Extract invoice fields from the provided PDF. "
        "Return ONLY valid JSON. No markdown. No extra text."
    )

    # Optional email context from Zapier (helps if PDF is messy)
    email_from = clamp_str(context.get("email_from"))
    email_subject = clamp_str(context.get("email_subject"))
    email_body = clamp_str(context.get("email_body"), limit=8000)

    user_prompt = f"""
Extract the following fields from the invoice PDF and return STRICT JSON with exactly these keys:
{{
  "invoice_number": string,
  "vendor": string,
  "invoice_date": string,   // ISO preferred: YYYY-MM-DD if possible
  "due_date": string,       // ISO preferred: YYYY-MM-DD if possible
  "currency": string,       // e.g., USD
  "subtotal": number,
  "tax": number,
  "total": number,
  "confidence": number,     // 0.0 to 1.0
  "flags": string[],        // any concerns: missing PO, bank detail change, unclear totals, etc.
  "summary": string         // 1–2 sentence human summary
}}

Rules:
- If a field is missing, use empty string for strings, 0 for numbers, [] for flags.
- Do not hallucinate. If unsure, lower confidence and add a flag explaining why.

Email context (may help; not authoritative):
- From: {email_from}
- Subject: {email_subject}
- Body (truncated): {email_body}
""".strip()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )

    raw_text = getattr(response, "text", "") or ""
    data = extract_json(raw_text)
    normalized = normalize_invoice_payload(data)

    if DEBUG:
        normalized["_debug_model_text"] = raw_text[:2000]

    return normalized


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME})


@app.post("/process_invoice")
def process_invoice():
    """
    Zapier should POST multipart/form-data:
      - invoice_pdf: file (application/pdf)
      - email_from: optional
      - email_subject: optional
      - email_body: optional
    """
    if "invoice_pdf" not in request.files:
        return jsonify({"error": "Missing file field 'invoice_pdf'"}), 400

    f = request.files["invoice_pdf"]
    pdf_bytes = f.read()

    if not pdf_bytes:
        return jsonify({"error": "Empty PDF upload"}), 400

    mb = len(pdf_bytes) / (1024 * 1024)
    if mb > MAX_PDF_MB:
        return jsonify({"error": f"PDF too large ({mb:.2f} MB). Limit is {MAX_PDF_MB} MB."}), 413

    context = {
        "email_from": request.form.get("email_from", ""),
        "email_subject": request.form.get("email_subject", ""),
        "email_body": request.form.get("email_body", ""),
    }

    try:
        result = gemini_extract_invoice(pdf_bytes, context)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Render will run via gunicorn, but local dev is fine:
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
