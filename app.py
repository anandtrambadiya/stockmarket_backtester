from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import date
from src.data_loader import load_data
from src.features import build_features
from src.visualizer import plot_features, plot_backtest
from src.model import train_model
from src.backtest import run_backtest
from src.reporter import generate_report

app = Flask(__name__)

@app.route('/')
def index():
    data = get_cached_data()
    return jsonify(data)


def run_pipeline():
    load_data()
    df = build_features()
    plot_features(df) 
    model, predictions, X_test, y_test, accuracy = train_model(df)
    results, sharpe, max_drawdown, win_rate = run_backtest(predictions, X_test, y_test, df)
    plot_backtest(results)  # generates backtest_plot.png
    report = generate_report(sharpe, max_drawdown, win_rate, accuracy, results["Cumulative_Strategy"].iloc[-1], results["Cumulative_Market"].iloc[-1])
    
    return {
        "date": str(date.today()),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "cumulative_strategy": results["Cumulative_Strategy"].iloc[-1],
        "cumulative_market": results["Cumulative_Market"].iloc[-1],
        "report": report
    }

def get_cached_data():
    # check if cache.json exists and is from today
    from pathlib import Path

    # File path
    a = Path("data/cache.json")

    # Check if the file exists
    if a.exists():
        try:
            with open("data/cache.json", "r") as file:
                cache = json.load(file)
            if cache["date"] == str(date.today()):
                return cache
            else:
                result = run_pipeline()
                with open("data/cache.json", "w") as file:
                    json.dump(result, file)
                return result
        except (json.JSONDecodeError, KeyError):
            result = run_pipeline()
            with open("data/cache.json", "w") as file:
                json.dump(result, file)
            return result
    else:
        result = run_pipeline()
        
        with open("data/cache.json", "w") as file:
            json.dump(result, file)
        return result
    

if __name__ == '__main__':
    app.run(debug=True)
    