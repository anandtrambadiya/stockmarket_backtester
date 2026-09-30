from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, jsonify, request
from src.earnings.dashboard_data import build_dashboard_data
from src.stock.engine import analyse
import os, json

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-120b"


# ── Home ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", active="home")


# ── Earnings ──────────────────────────────────────────────────────────────────

@app.route("/earnings")
def earnings():
    data = build_dashboard_data()
    return render_template("earnings.html", data=data, active="earnings")

@app.route("/api/earnings")
def api_earnings():
    return jsonify(build_dashboard_data())

@app.route("/api/earnings/refresh")
def api_earnings_refresh():
    data = build_dashboard_data(force=True)
    return jsonify({"status": "ok", "events": len(data["events"])})


# ── Stock ─────────────────────────────────────────────────────────────────────

@app.route("/stock")
def stock_home():
    return render_template("stock.html", data=None, active="stock")

@app.route("/stock/<ticker>")
def stock_detail(ticker):
    data = analyse(ticker.upper())
    return render_template("stock.html", data=data, ticker=ticker.upper(), active="stock")

@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    return jsonify(analyse(ticker.upper()))


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.route("/chat")
def chat():
    return render_template("chat.html", active="chat")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    import requests as req_lib
    key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not key:
        return jsonify({"error": "GROQ_API_KEY not set in .env"}), 500
    body     = request.get_json(force=True)
    messages = body.get("messages", [])
    system   = body.get("system", "You are a helpful quantitative analyst.")
    full_messages = [{"role": "system", "content": system}] + [
        m for m in messages if m.get("role") in ("user", "assistant")
    ]
    try:
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": full_messages, "max_tokens": 1024, "temperature": 0.7, "stream": False},
            timeout=30,
        )
        print(f"[Groq] status={resp.status_code}")
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                msg = resp.text
            return jsonify({"error": f"Groq {resp.status_code}: {msg}"}), 500
        reply = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"[Groq Exception] {e}")
        return jsonify({"error": str(e)}), 500


# ── Sarvam STT ────────────────────────────────────────────────────────────────

@app.route("/api/stt", methods=["POST"])
def api_stt():
    import requests as req_lib
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        return jsonify({"error": "SARVAM_API_KEY not set in .env"}), 500
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    lang = request.form.get("language_code", "unknown")
    try:
        resp = req_lib.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": key},
            files={"file": (audio_file.filename or "audio.webm", audio_file.read(), "audio/webm")},
            data={"model": "saaras:v3", "language_code": lang},
            timeout=30,
        )
        print(f"[Saaras STT] status={resp.status_code}")
        if resp.status_code != 200:
            return jsonify({"error": f"Saaras {resp.status_code}: {resp.text}"}), 500
        data = resp.json()
        return jsonify({"transcript": data.get("transcript", ""), "language_code": data.get("language_code", lang)})
    except Exception as e:
        print(f"[Saaras STT Exception] {e}")
        return jsonify({"error": str(e)}), 500


# ── Sarvam TTS ────────────────────────────────────────────────────────────────

@app.route("/api/tts", methods=["POST"])
def api_tts():
    import requests as req_lib
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        return jsonify({"error": "SARVAM_API_KEY not set in .env"}), 500
    body = request.get_json(force=True)
    text = body.get("text", "").strip()[:450]
    lang = body.get("language_code", "hi-IN")
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        resp = req_lib.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={"inputs": [text], "target_language_code": lang, "speaker": "anand", "model": "bulbul:v3", "enable_preprocessing": True},
            timeout=30,
        )
        print(f"[Bulbul TTS] status={resp.status_code}")
        if resp.status_code != 200:
            print(f"[Bulbul TTS] error body: {resp.text}")
            return jsonify({"error": f"Bulbul {resp.status_code}: {resp.text}"}), 500
        data   = resp.json()
        audios = data.get("audios", [])
        if not audios:
            return jsonify({"error": "No audio returned"}), 500
        return jsonify({"audio_b64": audios[0], "language_code": lang})
    except Exception as e:
        print(f"[Bulbul TTS Exception] {e}")
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)