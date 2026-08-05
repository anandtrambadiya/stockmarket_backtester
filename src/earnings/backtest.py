"""
Backtest the earnings drift strategy.
Only take trades where model prediction is in top 25%.
Apply 3% stop loss. Hold max 15 days.
"""

import pandas as pd
import numpy as np

BROKERAGE    = 0.001    # 0.1% round trip (entry + exit)
STOP_LOSS    = -3.0     # % — exit if stock drops 3% from entry

def run_earnings_backtest(test_df):
    """
    test_df must have: predicted, target_15d, entry_price, exit_price, stop_hit
    Only trades top 25% predicted returns.
    """
    df = test_df.copy().sort_index()

    # Filter — only top quartile confidence trades
    threshold  = np.percentile(df['predicted'], 75)
    trades     = df[df['predicted'] >= threshold].copy()

    print(f"\nTotal events in test period : {len(df)}")
    print(f"Trades taken (top 25%)      : {len(trades)}")

    # Apply stop loss
    trades['actual_return'] = np.where(
        trades['stop_hit'] == 1,
        STOP_LOSS / 100,
        (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    )

    # Deduct brokerage
    trades['actual_return'] -= BROKERAGE

    # Portfolio curve — assume equal capital per trade, 1 trade at a time
    trades['cumulative'] = (1 + trades['actual_return']).cumprod()

    # Benchmark — take ALL trades (no ML filter)
    all_trades = df.copy()
    all_trades['actual_return'] = np.where(
        all_trades['stop_hit'] == 1,
        STOP_LOSS / 100,
        (all_trades['exit_price'] - all_trades['entry_price']) / all_trades['entry_price']
    ) - BROKERAGE
    all_trades['cumulative'] = (1 + all_trades['actual_return']).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(t, label):
        ret      = t['actual_return']
        cum      = t['cumulative']
        sharpe   = ret.mean() / ret.std() * np.sqrt(252 / 15) if ret.std() > 0 else 0
        max_dd   = (cum / cum.cummax() - 1).min()
        win_rate = (ret > 0).mean()
        total    = cum.iloc[-1] - 1
        print(f"\n{label}")
        print(f"  Trades      : {len(t)}")
        print(f"  Total return: {total:.2%}")
        print(f"  Avg per trade: {ret.mean():.2%}")
        print(f"  Sharpe      : {sharpe:.3f}")
        print(f"  Max drawdown: {max_dd:.2%}")
        print(f"  Win rate    : {win_rate:.2%}")
        print(f"  Stop loss hit: {t['stop_hit'].mean():.2%}")

    print("\n" + "=" * 55)
    print("EARNINGS DRIFT BACKTEST")
    print("=" * 55)
    metrics(trades,     "ML-FILTERED (top 25% only)")
    metrics(all_trades, "BASELINE (all events, no ML filter)")

    print(f"\n  ML lift vs baseline: {trades['actual_return'].mean() - all_trades['actual_return'].mean():+.2%} per trade")

    return trades, all_trades