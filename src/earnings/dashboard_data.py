"""
Builds data for the earnings dashboard.
Computes per-event cards + aggregate stats.
Cached daily.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import date, timedelta
from .data_loader import load_ticker, build_events, download_all

CACHE_PATH = "data/earnings_cache.json"

def compute_drift_stats(ticker, event_date, df):
    """For one event, compute historical drift at 5/10/15 days."""
    if df is None:
        return {}
    try:
        loc = df.index.get_loc(event_date)
    except KeyError:
        return {}

    stats = {}
    for days in [5, 10, 15]:
        if loc + days < len(df):
            entry = df['Open'].iloc[loc + 1]
            exit_ = df['Close'].iloc[loc + days]
            stats[f'ret_{days}d'] = round((exit_ - entry) / entry * 100, 2)
        else:
            stats[f'ret_{days}d'] = None
    return stats


def historical_drift_profile(ticker, df, exclude_date):
    """
    For all past events of this ticker, compute avg drift at 5/10/15 days.
    Gives user a sense of 'how does this stock historically behave after big gaps'.
    """
    from .data_loader import detect_earnings_gaps
    if df is None or len(df) < 60:
        return {}

    past_events = detect_earnings_gaps(df, ticker)
    past_events  = past_events[past_events.index < exclude_date]

    if len(past_events) == 0:
        return {}

    drifts = {'5d': [], '10d': [], '15d': []}
    for ev_date in past_events.index:
        stats = compute_drift_stats(ticker, ev_date, df)
        for days, key in [(5,'5d'),(10,'10d'),(15,'15d')]:
            val = stats.get(f'ret_{days}d')
            if val is not None:
                drifts[key].append(val)

    result = {}
    for key, vals in drifts.items():
        if vals:
            result[f'hist_avg_{key}']      = round(np.mean(vals), 2)
            result[f'hist_positive_{key}'] = round((np.array(vals) > 0).mean() * 100, 1)
            result[f'hist_n']              = len(vals)
    return result


def signal_strength(gap_pct, vol_ratio, close_strength, hist_positive_15d=None):
    """Simple scoring: 0-100 → maps to star rating."""
    score = 0
    score += min(gap_pct / 10 * 40, 40)       # gap size: max 40pts at 10%
    score += min(vol_ratio / 10 * 30, 30)      # volume: max 30pts at 10x
    score += close_strength * 20               # close strength: max 20pts
    if hist_positive_15d:
        score += min((hist_positive_15d - 50) / 50 * 10, 10)  # history bonus: max 10pts
    score = max(0, min(100, score))
    if score >= 70:   stars = '★★★'
    elif score >= 45: stars = '★★☆'
    else:             stars = '★☆☆'
    return round(score, 1), stars


def build_dashboard_data(force=False):
    """Build full dashboard data. Cache for the day."""
    if os.path.exists(CACHE_PATH) and not force:
        try:
            cache = json.loads(open(CACHE_PATH).read())
            if cache.get('date') == str(date.today()):
                return cache
        except Exception:
            pass

    # Ensure data exists
    download_all()
    events_df = build_events()

    if events_df.empty:
        return {'date': str(date.today()), 'events': [], 'stats': {}}

    cards  = []
    all_15d = []

    for ev_date, row in events_df.iterrows():
        ticker = row['ticker']
        df     = load_ticker(ticker)
        if df is None:
            continue

        # forward drift for this event
        drift = compute_drift_stats(ticker, ev_date, df)

        # historical profile for this stock
        hist  = historical_drift_profile(ticker, df, ev_date)

        # signal strength
        score, stars = signal_strength(
            row['gap_pct'],
            row['vol_ratio'],
            row.get('close_strength', 0.7),
            hist.get('hist_positive_15d')
        )

        card = {
            'ticker'         : ticker.replace('.NS',''),
            'date'           : str(ev_date.date()),
            'gap_pct'        : round(row['gap_pct'], 2),
            'vol_ratio'      : round(row['vol_ratio'], 1),
            'close_strength' : round(row.get('close_strength', 0), 2),
            'signal_score'   : score,
            'signal_stars'   : stars,
            **drift,
            **hist,
        }
        cards.append(card)

        if drift.get('ret_15d') is not None:
            all_15d.append(drift['ret_15d'])

    # Sort by date descending
    cards.sort(key=lambda x: x['date'], reverse=True)

    # Aggregate stats
    ret_arr = np.array(all_15d)
    stats = {
        'total_events'     : len(cards),
        'avg_gap'          : round(float(events_df['gap_pct'].mean()), 2),
        'avg_vol_ratio'    : round(float(events_df['vol_ratio'].mean()), 1),
        'avg_15d_return'   : round(float(ret_arr.mean()), 2) if len(ret_arr) else 0,
        'pct_positive_15d' : round(float((ret_arr > 0).mean() * 100), 1) if len(ret_arr) else 0,
        'stocks_covered'   : int(events_df['ticker'].nunique()),
        'date_range_start' : str(events_df.index.min().date()),
        'date_range_end'   : str(events_df.index.max().date()),
    }

    data = {
        'date'  : str(date.today()),
        'events': cards,
        'stats' : stats,
    }

    os.makedirs('data', exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(data, f)

    return data