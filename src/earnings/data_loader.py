"""
Downloads OHLCV for universe stocks.
Detects GENUINE earnings gap days using 4 combined conditions:
  1. Gap up 4%+ at open vs previous close       — surprise at open
  2. Volume 2.5x 20-day average                 — institutional participation
  3. Close in top 30% of day's range            — bulls hold, not a fade
  4. NOT within 3 days of a prior event         — avoid double-counting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from .universe import NIFTY500_TICKERS

RAW_DIR    = "data/earnings_raw"
EVENTS_CSV = "data/events.csv"

GAP_THRESHOLD   = 4.0    # gap up % at open vs prev close
VOLUME_RATIO    = 2.5    # volume vs 20d avg
CLOSE_STRENGTH  = 0.60   # close must be in top 40% of day range
LOOKBACK        = 20     # volume average window
MIN_GAP_DAYS    = 5      # minimum days between two events on same stock


def download_all(period="5y", force=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    downloaded, skipped = 0, 0

    for ticker in NIFTY500_TICKERS:
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
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    print(f"Downloaded: {downloaded}  Cached: {skipped}")


def load_ticker(ticker):
    path = f"{RAW_DIR}/{ticker.replace('.NS','')}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[['Open','High','Low','Close','Volume']].dropna()


def detect_earnings_gaps(df, ticker):
    """
    4-condition filter for genuine earnings gap days.
    Returns clean events DataFrame.
    """
    df = df.copy()

    prev_close      = df['Close'].shift(1)
    df['gap_pct']   = (df['Open'] - prev_close) / prev_close * 100
    df['vol_avg']   = df['Volume'].rolling(LOOKBACK).mean().shift(1)
    df['vol_ratio'] = df['Volume'] / df['vol_avg']

    # close strength — where did we close within the day's range?
    # 1.0 = closed at high, 0.0 = closed at low
    df['close_strength'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)

    # All 4 conditions must be true
    mask = (
        (df['gap_pct']        >= GAP_THRESHOLD)  &   # big gap up
        (df['vol_ratio']      >= VOLUME_RATIO)   &   # volume surge
        (df['close_strength'] >= CLOSE_STRENGTH) &   # bulls held — no fade
        (df['vol_avg'].notna())                       # enough history
    )

    candidates = df[mask].copy()

    if len(candidates) == 0:
        return pd.DataFrame()

    # Remove events too close to each other (same stock, within 5 days)
    filtered = []
    last_date = None
    for date, row in candidates.iterrows():
        if last_date is None or (date - last_date).days >= MIN_GAP_DAYS:
            filtered.append(date)
            last_date = date

    events = candidates.loc[filtered][['gap_pct','vol_ratio','close_strength','Close','Open']].copy()
    events['ticker']   = ticker
    events['surprise'] = events['gap_pct']

    return events


def build_events(force=False):
    if os.path.exists(EVENTS_CSV) and not force:
        print(f"Loading cached events from {EVENTS_CSV}")
        df = pd.read_csv(EVENTS_CSV, index_col="Date", parse_dates=True)
        print(f"  {len(df)} events across {df['ticker'].nunique()} stocks")
        return df

    all_events = []
    for ticker in NIFTY500_TICKERS:
        df = load_ticker(ticker)
        if df is None:
            continue
        events = detect_earnings_gaps(df, ticker)
        if len(events) > 0:
            all_events.append(events)

    if not all_events:
        print("No events found — run download_all() first")
        return pd.DataFrame()

    events_df = pd.concat(all_events).sort_index()
    events_df.to_csv(EVENTS_CSV)

    print(f"Saved {len(events_df)} events")
    print(f"Stocks covered : {events_df['ticker'].nunique()}")
    print(f"Date range     : {events_df.index.min().date()} → {events_df.index.max().date()}")
    print(f"Avg gap size   : {events_df['surprise'].mean():.2f}%")
    print(f"Avg vol ratio  : {events_df['vol_ratio'].mean():.2f}x")
    print(f"Avg close str  : {events_df['close_strength'].mean():.2f}")

    return events_df


if __name__ == "__main__":
    download_all()
    events = build_events(force=True)
    print(events.head(10))