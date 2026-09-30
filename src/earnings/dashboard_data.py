"""
Dashboard data builder — feeds the /earnings page.
Loads saved model from models/, scores recent events, caches daily.
Falls back to rule-based signal score if no model exists yet.
"""

import pandas as pd
import numpy as np
import os, json
from datetime import date
from .data_loader import load_ticker, build_events, download_all, detect_events
from .features   import FEATURE_COLS

CACHE_PATH   = "data/earnings_cache.json"
MODEL_PATH   = "models/earnings_drift.json"
METRICS_PATH = "models/earnings_metrics.json"


def signal_score(gap_pct, vol_ratio, close_strength, hist_positive=None):
    """Rule-based score 0–100 → star rating. Used when ML model not trained yet."""
    score  = min(gap_pct / 8 * 40, 40)
    score += min(vol_ratio / 8 * 30, 30)
    score += close_strength * 20
    if hist_positive:
        score += min((hist_positive - 50) / 50 * 10, 10)
    score = max(0, min(100, score))
    stars = "★★★" if score >= 70 else ("★★☆" if score >= 45 else "★☆☆")
    return round(score, 1), stars


def ml_score(model, event_row, nifty_df, ticker_df, event_date):
    """Score a single event using the trained model. Returns predicted 10d return."""
    try:
        loc = ticker_df.index.get_loc(event_date)
        if loc < 20:
            return None
        pre  = ticker_df.iloc[loc-20:loc]
        day0 = ticker_df.iloc[loc]

        # Build the same features as training
        vix = nifty_ret = np.nan
        if nifty_df is not None and event_date in nifty_df.index:
            vix     = float(nifty_df.loc[event_date, "vix_close"])
            nifty_ret = float(nifty_df["nifty_close"].pct_change(5).loc[event_date] * 100)

        row = {
            "surprise_pct"  : float(event_row.get("gap_pct", 0)),
            "volume_ratio"  : float(event_row.get("vol_ratio", 1)),
            "close_strength": float(event_row.get("close_str", 0.7)),
            "pre_mom_5d"    : float((pre["Close"].iloc[-1] / pre["Close"].iloc[-5] - 1) * 100),
            "pre_mom_20d"   : float((pre["Close"].iloc[-1] / pre["Close"].iloc[0]  - 1) * 100),
            "pre_vol"       : float(pre["Close"].pct_change().std() * np.sqrt(252) * 100),
            "gap_open"      : float((day0["Open"] - pre["Close"].iloc[-1]) / pre["Close"].iloc[-1] * 100),
            "ret_day0"      : float((day0["Close"] - day0["Open"]) / day0["Open"] * 100),
            "vix_level"     : vix,
            "nifty_5d_ret"  : nifty_ret,
            "day_of_week"   : float(event_date.dayofweek),
        }
        X = pd.DataFrame([row])[FEATURE_COLS]
        return float(model.predict(X)[0])
    except Exception:
        return None


def drift_at_days(ticker_df, event_date, days_list=(5, 10)):
    """Actual forward return at N days after event (for completed events)."""
    if ticker_df is None:
        return {}
    try:
        loc = ticker_df.index.get_loc(event_date)
    except KeyError:
        return {}
    stats = {}
    for d in days_list:
        if loc + d < len(ticker_df):
            entry = ticker_df["Open"].iloc[loc + 1]
            exit_ = ticker_df["Close"].iloc[loc + d]
            stats[f"ret_{d}d"] = round((exit_ - entry) / entry * 100, 2)
        else:
            stats[f"ret_{d}d"] = None
    return stats


def historical_drift(ticker, ticker_df, exclude_date):
    """Avg drift across all past events for this stock."""
    if ticker_df is None or len(ticker_df) < 60:
        return {}
    past = detect_events(ticker_df, ticker)
    past = past[past.index < exclude_date]
    if len(past) == 0:
        return {}
    vals = []
    for d in past.index:
        s = drift_at_days(ticker_df, d, [10])
        if s.get("ret_10d") is not None:
            vals.append(s["ret_10d"])
    if not vals:
        return {}
    arr = np.array(vals)
    return {
        "hist_avg_10d"     : round(float(arr.mean()), 2),
        "hist_positive_10d": round(float((arr > 0).mean() * 100), 1),
        "hist_n"           : len(vals),
    }


def build_dashboard_data(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        try:
            cache = json.loads(open(CACHE_PATH).read())
            if cache.get("date") == str(date.today()):
                return cache
        except Exception:
            pass

    download_all()
    events_df = build_events()
    if events_df.empty:
        return {"date": str(date.today()), "events": [], "stats": {}, "model_trained": False}

    # Load ML model if available
    model = None
    model_meta = {}
    if os.path.exists(MODEL_PATH):
        from .model import load_model
        try:
            model = load_model()
        except Exception:
            pass
    if os.path.exists(METRICS_PATH):
        model_meta = json.loads(open(METRICS_PATH).read())

    # Load Nifty/VIX context
    nifty_df = None
    if os.path.exists("data/gap_raw.csv"):
        nifty_df = pd.read_csv("data/gap_raw.csv", index_col="Date", parse_dates=True)

    cards = []
    all_10d = []

    for ev_date, row in events_df.iterrows():
        ticker    = row["ticker"]
        ticker_df = load_ticker(ticker)
        if ticker_df is None:
            continue

        drift = drift_at_days(ticker_df, ev_date, [5, 10])
        hist  = historical_drift(ticker, ticker_df, ev_date)

        # ML rank score
        ml_pred = None
        if model is not None:
            ml_pred = ml_score(model, row, nifty_df, ticker_df, ev_date)

        score, stars = signal_score(
            row["gap_pct"], row["vol_ratio"],
            row.get("close_str", 0.7),
            hist.get("hist_positive_10d")
        )

        card = {
            "ticker"        : ticker,
            "date"          : str(ev_date.date()),
            "gap_pct"       : round(float(row["gap_pct"]), 2),
            "vol_ratio"     : round(float(row["vol_ratio"]), 1),
            "close_strength": round(float(row.get("close_str", 0)), 2),
            "signal_score"  : score,
            "signal_stars"  : stars,
            "ml_pred_10d"   : round(ml_pred, 2) if ml_pred is not None else None,
            **drift,
            **hist,
        }
        cards.append(card)
        if drift.get("ret_10d") is not None:
            all_10d.append(drift["ret_10d"])

    cards.sort(key=lambda x: x["date"], reverse=True)
    ret_arr = np.array(all_10d) if all_10d else np.array([0])

    stats = {
        "total_events"    : len(cards),
        "avg_gap"         : round(float(events_df["gap_pct"].mean()), 2),
        "avg_vol_ratio"   : round(float(events_df["vol_ratio"].mean()), 1),
        "avg_10d_return"  : round(float(ret_arr.mean()), 2),
        "pct_positive_10d": round(float((ret_arr > 0).mean() * 100), 1),
        "stocks_covered"  : int(events_df["ticker"].nunique()),
        "date_from"       : str(events_df.index.min().date()),
        "date_to"         : str(events_df.index.max().date()),
    }

    data = {
        "date"         : str(date.today()),
        "events"       : cards,
        "stats"        : stats,
        "model_trained": model is not None,
        "model_meta"   : model_meta,
    }

    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)

    return data
