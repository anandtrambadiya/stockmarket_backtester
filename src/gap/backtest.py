import pandas as pd
import numpy as np

BROKERAGE = 0.0005   # 0.05% per trade — STT + brokerage rough estimate

def run_gap_backtest(feat, predictions, X_test):
    """
    Trade logic:
      Predict fill  + gap up   → SHORT open, cover close  (fade the gap up)
      Predict fill  + gap down → BUY open, sell close     (fade the gap down)
      Predict extend + gap up   → BUY open, sell close    (follow gap up)
      Predict extend + gap down → SHORT open, cover close (follow gap down)

    Returns are computed on actual nifty_open → nifty_close, not gap_pct proxy.
    """
    results = feat.loc[X_test.index].copy()
    results['predicted'] = predictions

    gap_up         = results['gap_pct'] > 0
    fill_predicted = results['predicted'] == 1

    # trade direction: +1 = long, -1 = short
    results['trade_dir'] = np.where(
        (fill_predicted & gap_up) | (~fill_predicted & ~gap_up), -1, 1
    )

    # actual intraday return: (close - open) / open
    intraday_ret = (results['nifty_close'] - results['nifty_open']) / results['nifty_open']

    # strategy return = trade direction × intraday return − cost
    results['daily_returns'] = results['trade_dir'] * intraday_ret - BROKERAGE

    results['cumulative']        = (1 + results['daily_returns']).cumprod()
    results['cumulative_market'] = (1 + intraday_ret).cumprod()   # buy-and-hold baseline

    # ── Metrics ───────────────────────────────────────────────────────────────
    ann_factor  = np.sqrt(252)
    sharpe      = results['daily_returns'].mean() / results['daily_returns'].std() * ann_factor
    max_dd      = (results['cumulative'] / results['cumulative'].cummax() - 1).min()
    win_rate    = (results['daily_returns'] > 0).mean()
    total_ret   = results['cumulative'].iloc[-1] - 1
    mkt_ret     = results['cumulative_market'].iloc[-1] - 1

    print("\n" + "=" * 50)
    print("GAP BACKTEST RESULTS  (actual open→close returns)")
    print("=" * 50)
    print(f"  Total trades    : {len(results)}")
    print(f"  Strategy return : {total_ret:.2%}")
    print(f"  Market return   : {mkt_ret:.2%}  (buy-and-hold baseline)")
    print(f"  Sharpe ratio    : {sharpe:.3f}")
    print(f"  Max drawdown    : {max_dd:.2%}")
    print(f"  Win rate        : {win_rate:.2%}")
    print(f"  Brokerage paid  : {BROKERAGE * len(results):.2%}  total")

    return results, sharpe, max_dd, win_rate