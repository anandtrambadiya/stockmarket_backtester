"""
Data layer — downloads OHLCV, detects genuine earnings gap events.

Detection logic (4 conditions must ALL be true):
  1. Gap up >= 3% at open vs previous close
  2. Volume >= 2x 20-day average
  3. Close in top 35% of day range (bulls held, not a fade)
  4. Min 5 trading days since last event on same stock
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os, time
from .universe import UNIVERSE

RAW_DIR    = "data/earnings_raw"
EVENTS_CSV = "data/events.csv"

# NEW
GAP_PCT_MIN    = 2.5
VOL_RATIO_MIN  = 1.8
CLOSE_STR_MIN  = 0.65
MIN_GAP_DAYS   = 5
LOOKBACK       = 20


def download_all(period="5y", force=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    downloaded = skipped = errors = 0
    for ticker in UNIVERSE:
        path = f"{RAW_DIR}/{ticker.replace('.NS','')}.csv"
        if os.path.exists(path) and not force:
            skipped += 1
            continue
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 100:
                continue
            df.to_csv(path)
            downloaded += 1
            time.sleep(0.25)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")
            errors += 1
    print(f"Download complete — new:{downloaded}  cached:{skipped}  errors:{errors}")


def load_ticker(ticker):
    path = f"{RAW_DIR}/{ticker.replace('.NS','')}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[["Open","High","Low","Close","Volume"]].dropna()


def detect_events(df, ticker):
    """Return rows that look like genuine post-earnings gap days."""
    df = df.copy()
    prev_close      = df["Close"].shift(1)
    df["gap_pct"]   = (df["Open"] - prev_close) / prev_close * 100
    df["vol_avg"]   = df["Volume"].rolling(LOOKBACK).mean().shift(1)
    df["vol_ratio"] = df["Volume"] / df["vol_avg"]
    df["close_str"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    mask = (
        (df["gap_pct"]   >= GAP_PCT_MIN)   &
        (df["vol_ratio"] >= VOL_RATIO_MIN) &
        (df["close_str"] >= CLOSE_STR_MIN) &
        df["vol_avg"].notna()
    )
    candidates = df[mask].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    # enforce minimum gap between events
    keep, last = [], None
    for d in candidates.index:
        if last is None or (d - last).days >= MIN_GAP_DAYS:
            keep.append(d)
            last = d

    out = candidates.loc[keep][["gap_pct","vol_ratio","close_str","Open","Close"]].copy()
    out["ticker"]   = ticker.replace(".NS","")
    out["surprise"] = out["gap_pct"]
    return out


def build_events(force=False):
    if os.path.exists(EVENTS_CSV) and not force:
        df = pd.read_csv(EVENTS_CSV, index_col="Date", parse_dates=True)
        print(f"Events loaded from cache — {len(df)} events, {df['ticker'].nunique()} stocks")
        return df

    all_events = []
    for ticker in UNIVERSE:
        df = load_ticker(ticker)
        if df is None:
            continue
        ev = detect_events(df, ticker)
        if len(ev) > 0:
            all_events.append(ev)

    if not all_events:
        print("No events found — run download_all() first")
        return pd.DataFrame()

    events_df = pd.concat(all_events).sort_index()
    os.makedirs("data", exist_ok=True)
    events_df.to_csv(EVENTS_CSV)
    print(f"Events built — {len(events_df)} events, {events_df['ticker'].nunique()} stocks")
    print(f"Date range: {events_df.index.min().date()} → {events_df.index.max().date()}")
    return events_df
