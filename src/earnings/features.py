"""
Feature engineering for earnings drift model.

Target: 10-day forward return from next-day open (realistic entry).
We rank by predicted return, not classify direction.

Features chosen because they have documented predictive power
for post-earnings drift specifically:
  - Surprise size & volume (event strength)
  - Pre-event momentum & volatility (context)
  - Market regime (Nifty 500 50-day trend)
  - Day-of-week (Thursday = F&O expiry = different behavior)

NOTE: Bear market events are NOT removed from training.
The model needs to SEE bear markets to learn to avoid them.
The hard nifty_trend > 0 filter is applied at backtest time only.
"""

import pandas as pd
import numpy as np
import os
import yfinance as yf
from .data_loader import load_ticker

VIX_CSV = "data/gap_raw.csv"

FEATURE_COLS = [
    "surprise_pct",     # gap size on results day
    "volume_ratio",     # volume vs 20d avg
    "close_strength",   # where close landed in day range (1=top)
    "pre_mom_5d",       # stock return 5 days before event
    "pre_mom_20d",      # stock return 20 days before event
    "pre_vol",          # 20d realized vol before event (annualised %)
    "gap_open",         # open gap vs prev close
    "ret_day0",         # event day intraday return (open->close)
    "day_of_week",      # 0=Mon ... 4=Fri
    "nifty_trend",      # Nifty 500 50-day return -- bull/bear regime
]

MACRO_COLS = ["vix_level", "nifty_5d_ret"]


def _load_nifty500():
    """Download Nifty 500 index once for regime detection."""
    try:
        raw = yf.download("^CRSLDX", period="6y", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw["Close"].dropna()
    except Exception:
        return None


def _nifty_trend(nifty_close, date, window=50):
    """50-day return of Nifty 500 at event date."""
    if nifty_close is None:
        return 0.0
    try:
        loc = nifty_close.index.get_indexer([date], method="nearest")[0]
        if loc < window:
            return 0.0
        return float(
            (nifty_close.iloc[loc] / nifty_close.iloc[loc - window] - 1) * 100
        )
    except Exception:
        return 0.0


def build_features(events_df, nifty_df=None):
    if nifty_df is None and os.path.exists(VIX_CSV):
        nifty_df = pd.read_csv(VIX_CSV, index_col="Date", parse_dates=True)

    print("  Downloading Nifty 500 for regime filter...")
    nifty_close = _load_nifty500()
    if nifty_close is not None:
        print(f"  Nifty 500 loaded -- {len(nifty_close)} days")
    else:
        print("  Nifty 500 unavailable -- nifty_trend will be 0.0 for all events")

    rows = []
    skipped = 0

    for date, event in events_df.iterrows():
        ticker = event["ticker"]
        df = load_ticker(ticker)
        if df is None or len(df) < 60:
            skipped += 1
            continue
        if date not in df.index:
            skipped += 1
            continue

        loc = df.index.get_loc(date)
        if loc < 20 or loc + 11 >= len(df):
            skipped += 1
            continue

        pre  = df.iloc[loc-20 : loc]
        post = df.iloc[loc+1  : loc+11]
        day0 = df.iloc[loc]

        feat = {}
        feat["surprise_pct"]   = float(event["gap_pct"])
        feat["volume_ratio"]   = float(event["vol_ratio"])
        feat["close_strength"] = float(event.get("close_str", 0.7))

        feat["pre_mom_5d"]  = float((pre["Close"].iloc[-1] / pre["Close"].iloc[-5] - 1) * 100)
        feat["pre_mom_20d"] = float((pre["Close"].iloc[-1] / pre["Close"].iloc[0]  - 1) * 100)
        feat["pre_vol"]     = float(pre["Close"].pct_change().std() * np.sqrt(252) * 100)

        feat["gap_open"]    = float((day0["Open"] - pre["Close"].iloc[-1]) / pre["Close"].iloc[-1] * 100)
        feat["ret_day0"]    = float((day0["Close"] - day0["Open"]) / day0["Open"] * 100)
        feat["day_of_week"] = float(date.dayofweek)
        feat["nifty_trend"] = _nifty_trend(nifty_close, date, window=50)

        if nifty_df is not None and date in nifty_df.index:
            try:
                feat["vix_level"]    = float(nifty_df.loc[date, "vix_close"])
                feat["nifty_5d_ret"] = float(nifty_df["nifty_close"].pct_change(5).loc[date] * 100)
            except Exception:
                feat["vix_level"]    = np.nan
                feat["nifty_5d_ret"] = np.nan
        else:
            feat["vix_level"]    = np.nan
            feat["nifty_5d_ret"] = np.nan

        entry_price = float(post["Open"].iloc[0])
        exit_price  = float(post["Close"].iloc[-1])
        feat["target_10d"] = (exit_price - entry_price) / entry_price * 100

        post_low = float(post["Low"].min())
        feat["stop_hit"] = int((post_low - entry_price) / entry_price * 100 < -6.0)

        feat["entry_price"] = entry_price
        feat["exit_price"]  = exit_price
        feat["ticker"]      = ticker
        feat["date"]        = date

        rows.append(feat)

    feat_df = pd.DataFrame(rows).set_index("date")

    # Add macro cols only if they have real values
    has_macro = feat_df["vix_level"].notna().any()
    if has_macro:
        all_cols = FEATURE_COLS + MACRO_COLS
        print("  Macro features (VIX, Nifty): ENABLED")
    else:
        all_cols = FEATURE_COLS
        feat_df.drop(columns=MACRO_COLS, errors="ignore", inplace=True)
        print("  Macro features (VIX, Nifty): skipped -- gap_raw.csv not found")

    import src.earnings.features as _self
    _self.FEATURE_COLS = all_cols

    feat_df.dropna(subset=all_cols, inplace=True)

    print(f"\nFeature dataset: {len(feat_df)} events  ({skipped} skipped)")
    print(f"Stocks         : {feat_df['ticker'].nunique()}")
    print(f"Date range     : {feat_df.index.min().date()} --> {feat_df.index.max().date()}")
    print(f"Target 10d -- mean:{feat_df['target_10d'].mean():.2f}%  "
          f"median:{feat_df['target_10d'].median():.2f}%  "
          f"positive:{(feat_df['target_10d']>0).mean():.1%}")
    print(f"Stop hit       : {feat_df['stop_hit'].mean():.1%} of events")
    print(f"Avg nifty_trend: {feat_df['nifty_trend'].mean():.2f}%  "
          f"(bull events: {(feat_df['nifty_trend']>0).mean():.1%})")
    return feat_df