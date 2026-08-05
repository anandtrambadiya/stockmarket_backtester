import pandas as pd
import numpy as np

FEATURE_COLS = [
    'gap_pct',          # size of today's gap
    'gap_direction',    # +1 gap up, -1 gap down
    'sp500_ret',        # S&P 500 previous day return — overnight global sentiment
    'vix_level',        # India VIX level — fear gauge
    'vix_change',       # VIX change vs yesterday — rising fear or falling
    'usdinr_ret',       # USD/INR move — INR weakening = negative for Nifty
    'nifty_ret_1d',     # Nifty previous day return — momentum/mean reversion context
    'nifty_ret_5d',     # Nifty 5-day return — short trend context
    'day_of_week',      # 0=Mon, 3=Thu (F&O expiry), 4=Fri
    'gap_abs',          # absolute gap size — large gaps behave differently
]

def build_gap_features(df):
    """
    Build gap predictor features from raw OHLC data.
    Target: did the gap fill by close?
      - Gap up   → fill = Close < Open  (price came back down)
      - Gap down → fill = Close > Open  (price bounced back up)

    Only uses last 5 years — pre-2020 data is a different regime.
    """
    # ── Filter to last 5 years ────────────────────────────────────────────────
    cutoff = df.index.max() - pd.DateOffset(years=5)
    df     = df[df.index >= cutoff].copy()
    print(f"Using data from {df.index.min().date()} to {df.index.max().date()}")

    feat = pd.DataFrame(index=df.index)

    # carry actual open/close for backtest P&L — NOT used as model features
    feat['nifty_open']  = df['nifty_open']
    feat['nifty_close'] = df['nifty_close']

    # ── Gap features ─────────────────────────────────────────────────────────
    prev_close       = df['nifty_close'].shift(1)
    feat['gap_pct']  = (df['nifty_open'] - prev_close) / prev_close * 100
    feat['gap_abs']  = feat['gap_pct'].abs()
    feat['gap_direction'] = np.sign(feat['gap_pct'])

    # ── External features (all shifted 1 so no lookahead) ────────────────────
    # S&P 500 — previous day's return (closes before Nifty opens next day)
    feat['sp500_ret']  = df['sp500_close'].pct_change(1).shift(1) * 100

    # India VIX
    feat['vix_level']  = df['vix_close'].shift(1)
    feat['vix_change'] = df['vix_close'].pct_change(1).shift(1) * 100

    # USD/INR — INR weakening (number goes up) = bad for Nifty
    feat['usdinr_ret'] = df['usdinr_close'].pct_change(1).shift(1) * 100

    # Nifty momentum context
    feat['nifty_ret_1d'] = df['nifty_close'].pct_change(1).shift(1) * 100
    feat['nifty_ret_5d'] = df['nifty_close'].pct_change(5).shift(1) * 100

    # Calendar
    feat['day_of_week'] = df.index.dayofweek.astype(float)

    # ── Target ───────────────────────────────────────────────────────────────
    gap_up   = df['nifty_open'] > prev_close
    gap_fill = (gap_up  & (df['nifty_close'] < df['nifty_open'])) | \
               (~gap_up & (df['nifty_close'] > df['nifty_open']))

    feat['target'] = gap_fill.astype(int)

    # Only keep rows where gap is meaningful (> 0.1%) — tiny gaps are noise
    feat = feat[feat['gap_abs'] > 0.1]

    feat.dropna(inplace=True)

    print(f"Gap dataset: {len(feat)} rows")
    print(f"Gap fill rate: {feat['target'].mean():.2%}  (base rate)")
    print(f"Avg gap size : {feat['gap_abs'].mean():.3f}%")
    print(f"\nGap up fill rate  : {feat[feat['gap_direction']==1]['target'].mean():.2%}")
    print(f"Gap down fill rate: {feat[feat['gap_direction']==-1]['target'].mean():.2%}")

    return feat


if __name__ == "__main__":
    from data_loader import load_gap_data
    raw = load_gap_data()
    feat = build_gap_features(raw)
    print(feat.tail(5))