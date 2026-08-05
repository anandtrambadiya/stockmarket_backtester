from flask import Flask, render_template, jsonify, request, send_file
import json, os
from datetime import date
from pathlib import Path
from src.data_loader import load_data
from src.features import build_features
from src.visualizer import plot_features, plot_backtest
from src.model import train_model
from src.backtest import run_backtest
from src.reporter import generate_report
from src.earnings.dashboard_data import build_dashboard_data
from src.stock.engine import analyse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# ── Pipeline & caching ────────────────────────────────────────────────────────

def run_pipeline():
    load_data()
    df = build_features()
    plot_features(df)
    model, predictions, X_test, y_test, accuracy, metrics_df = train_model(df)
    results, sharpe, max_drawdown, win_rate = run_backtest(predictions, X_test, y_test, df)
    plot_backtest(results)
    report = generate_report(
        sharpe, max_drawdown, win_rate, accuracy,
        results["Cumulative_Strategy"].iloc[-1],
        results["Cumulative_Market"].iloc[-1]
    )
    return {
        "date": str(date.today()),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "cumulative_strategy": results["Cumulative_Strategy"].iloc[-1],
        "cumulative_market": results["Cumulative_Market"].iloc[-1],
        "report": report,
    }

def get_cached_data():
    cache_path = Path("data/cache.json")
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            if cache.get("date") == str(date.today()):
                return cache
        except (json.JSONDecodeError, KeyError):
            pass
    result = run_pipeline()
    cache_path.write_text(json.dumps(result))
    return result

# ── HTML routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', active='home')

@app.route('/dashboard')
def dashboard():
    data = get_cached_data()
    return render_template('dashboard.html', data=data, active='dashboard')

@app.route('/chat')
def chat():
    data = get_cached_data()
    return render_template('chat.html', data=data, active='chat')

# ── Data/chart routes ─────────────────────────────────────────────────────────

@app.route('/api/data')
def api_data():
    return jsonify(get_cached_data())

@app.route('/chart/backtest')
def chart_backtest():
    return send_file('data/backtest_plot.png', mimetype='image/png')

@app.route('/chart/features')
def chart_features():
    return send_file('data/features_plot.png', mimetype='image/png')

# ── Chat API ──────────────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def api_chat():
    body = request.get_json(force=True)
    messages = body.get('messages', [])
    system = body.get('system', 'You are a quantitative trading assistant.')

    api_key = os.getenv('GROQ_API_KEY')
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        messages=[{"role": "system", "content": system}] + messages,
        model="llama-3.1-8b-instant",
        max_tokens=400,
    )
    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})

# ── Stock analysis routes ─────────────────────────────────────────────────────

@app.route('/stock')
def stock_home():
    return render_template('stock.html', data=None, active='earnings')

@app.route('/stock/<ticker>')
def stock_detail(ticker):
    data = analyse(ticker.upper())
    return render_template('stock.html', data=data, ticker=ticker.upper(), active='earnings')

@app.route('/api/stock/<ticker>')
def api_stock(ticker):
    return jsonify(analyse(ticker.upper()))

# ── Earnings Intelligence routes ──────────────────────────────────────────────

@app.route('/earnings')
def earnings():
    data = build_dashboard_data()
    return render_template('earnings.html', data=data, active='earnings')

@app.route('/api/earnings')
def api_earnings():
    data = build_dashboard_data()
    return jsonify(data)

@app.route('/api/earnings/refresh')
def api_earnings_refresh():
    data = build_dashboard_data(force=True)
    return jsonify({'status': 'ok', 'events': len(data['events'])})

if __name__ == '__main__':
    app.run(debug=True)