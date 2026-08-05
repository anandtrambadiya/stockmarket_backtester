"""
Build features for each earnings event.
For each results day, we look at:
- How big was the surprise (price move + volume)
- What was the market context (VIX, Nifty trend)
- Stock-specific context (momentum, volatility, market cap proxy)

Target: 15-day forward return after entry (next day open)
"""

import pandas as pd
import numpy as np
import os
from .data_loader import load_ticker, NIFTY500_TICKERS

VIX_CSV = "data/gap_raw.csv"   # reuse VIX from gap project

FEATURE_COLS = [
    'surprise_pct',      # results day price move %
    'volume_ratio',      # volume vs 20d avg
    'pre_momentum_5d',   # stock return 5 days before results
    'pre_momentum_20d',  # stock return 20 days before results
    'pre_volatility',    # 20d realized vol before results
    'vix_level',         # India VIX on results day
    'nifty_5d_ret',      # Nifty 5d return before results (market trend)
    'gap_open',          # how much stock gapped at open on results day
    'ret_day0',          # full results day return (open to close)
    'day_of_week',       # Thursday = F&O expiry, different behavior
]

def build_features(events_df, nifty_df=None):
    """
    For each event in events_df, pull 30 days of pre-event data
    and compute features. Target = 15-day forward return.
    """
    # Load VIX + Nifty from gap data
    if nifty_df is None and os.path.exists(VIX_CSV):
        nifty_df = pd.read_csv(VIX_CSV, index_col="Date", parse_dates=True)

    rows = []

    for date, event in events_df.iterrows():
        ticker = event['ticker']
        df     = load_ticker(ticker)
        if df is None or len(df) < 60:
            continue

        # locate results day in stock data
        if date not in df.index:
            continue

        loc = df.index.get_loc(date)

        # need at least 20 days before and 16 days after
        if loc < 20 or loc + 16 >= len(df):
            continue

        pre  = df.iloc[loc-20 : loc]    # 20 days before results
        post = df.iloc[loc+1  : loc+16] # 15 days after results (the trade)
        day0 = df.iloc[loc]              # results day itself

        # ── Features ──────────────────────────────────────────────────────────
        feat = {}

        feat['surprise_pct']   = event['surprise']
        feat['volume_ratio']   = event['vol_ratio']

        feat['pre_momentum_5d']  = (pre['Close'].iloc[-1] / pre['Close'].iloc[-5] - 1) * 100
        feat['pre_momentum_20d'] = (pre['Close'].iloc[-1] / pre['Close'].iloc[0]  - 1) * 100
        feat['pre_volatility']   = pre['Close'].pct_change().std() * np.sqrt(252) * 100

        feat['gap_open']  = (day0['Open'] - pre['Close'].iloc[-1]) / pre['Close'].iloc[-1] * 100
        feat['ret_day0']  = (day0['Close'] - day0['Open']) / day0['Open'] * 100
        feat['day_of_week'] = float(date.dayofweek)

        # VIX + Nifty context
        if nifty_df is not None and date in nifty_df.index:
            feat['vix_level']   = nifty_df.loc[date, 'vix_close']
            feat['nifty_5d_ret'] = nifty_df['nifty_close'].pct_change(5).loc[date] * 100
        else:
            feat['vix_level']    = np.nan
            feat['nifty_5d_ret'] = np.nan

        # ── Target: 15-day forward return (entry = next open) ─────────────────
        entry_price = post['Open'].iloc[0]    # next day open — realistic entry
        exit_price  = post['Close'].iloc[-1]  # day 15 close
        feat['target_15d'] = (exit_price - entry_price) / entry_price * 100

        # also store for backtest
        feat['entry_price'] = entry_price
        feat['exit_price']  = exit_price
        feat['ticker']      = ticker
        feat['date']        = date

        # stop loss check — did stock hit -3% at any point in hold period?
        post_low = post['Low'].min()
        feat['stop_hit'] = int((post_low - entry_price) / entry_price * 100 < -3.0)

        rows.append(feat)

    feat_df = pd.DataFrame(rows).set_index('date')
    feat_df.dropna(subset=FEATURE_COLS, inplace=True)

    print(f"\nFeature dataset: {len(feat_df)} events")
    print(f"Stocks         : {feat_df['ticker'].nunique()}")
    print(f"Date range     : {feat_df.index.min().date()} → {feat_df.index.max().date()}")
    print(f"\nTarget (15d return) stats:")
    print(f"  Mean   : {feat_df['target_15d'].mean():.2f}%")
    print(f"  Median : {feat_df['target_15d'].median():.2f}%")
    print(f"  Positive: {(feat_df['target_15d'] > 0).mean():.1%}")
    print(f"  Stop hit: {feat_df['stop_hit'].mean():.1%} of trades")

    return feat_df