"""
Clean backtest -- long-only, top-25% ML filter, 6% stop loss.
No shorting. No leverage. Realistic.

Runs on the FULL prediction dataset (all folds combined).
Bear market events filtered at trade time (nifty_trend > 0).
Model still trains on bear events so it learns to recognise them.
"""

import pandas as pd
import numpy as np
import os, json

BROKERAGE  = 0.001   # 0.1% round trip (Zerodha delivery)
STOP_LOSS  = -6.0    # % hard stop -- matches features.py stop_hit threshold

REPORT_PATH = "reports/backtest_summary.json"


def run_backtest(test_df, full_feat_df=None):
    # Use full dataset if available, else fall back to last fold
    if full_feat_df is not None and "predicted" in full_feat_df.columns:
        df = full_feat_df.copy().sort_index()
        print(f"  Running backtest on FULL dataset ({len(df)} events, all folds)")
    else:
        df = test_df.copy().sort_index()
        print(f"  Running backtest on last fold only ({len(df)} events)")

    # Hard rule: only trade in bull markets
    # Model trained on bear events to learn the pattern, but we never trade them
    if "nifty_trend" in df.columns:
        before = len(df)
        df = df[df["nifty_trend"] > 0].copy()
        print(f"  Bear-market filter: {before - len(df)} events removed "
              f"-> {len(df)} bull-market events remain")

    threshold  = np.percentile(df["predicted"], 75)
    trades_ml  = df[df["predicted"] >= threshold].copy()
    trades_all = df.copy()

    def calc(trades, label):
        t = trades.copy()
        t["ret"] = np.where(
            t["stop_hit"] == 1,
            STOP_LOSS / 100,
            (t["exit_price"] - t["entry_price"]) / t["entry_price"]
        ) - BROKERAGE

        t["equity"] = (1 + t["ret"]).cumprod()

        if len(t) == 0:
            return t, {}

        sharpe   = t["ret"].mean() / t["ret"].std() * np.sqrt(252 / 10) if t["ret"].std() > 0 else 0
        max_dd   = float((t["equity"] / t["equity"].cummax() - 1).min())
        win_rate = float((t["ret"] > 0).mean())
        total    = float(t["equity"].iloc[-1] - 1)
        avg_ret  = float(t["ret"].mean())

        print(f"\n{label}")
        print(f"  Trades      : {len(t)}")
        print(f"  Total return: {total:.2%}")
        print(f"  Avg/trade   : {avg_ret:.2%}")
        print(f"  Sharpe      : {sharpe:.3f}")
        print(f"  Max drawdown: {max_dd:.2%}")
        print(f"  Win rate    : {win_rate:.2%}")
        print(f"  Stop hit    : {t['stop_hit'].mean():.2%}")

        return t, {
            "label"        : label,
            "trades"       : len(t),
            "total_return" : round(total * 100, 2),
            "avg_per_trade": round(avg_ret * 100, 3),
            "sharpe"       : round(sharpe, 3),
            "max_drawdown" : round(max_dd * 100, 2),
            "win_rate"     : round(win_rate * 100, 1),
            "stop_hit_pct" : round(float(t["stop_hit"].mean()) * 100, 1),
        }

    print("=" * 58)
    print("BACKTEST -- Long-only, 6% stop, top-25% ML filter")
    print("=" * 58)

    ml_trades,  ml_summary  = calc(trades_ml,  "ML-FILTERED (top 25% only)")
    all_trades, all_summary = calc(trades_all, "BASELINE (all bull-market events)")

    ml_lift = ml_summary.get("avg_per_trade", 0) - all_summary.get("avg_per_trade", 0)
    print(f"\n  ML lift vs baseline: {ml_lift:+.3f}% per trade")

    os.makedirs("reports", exist_ok=True)
    report = {
        "ml_strategy" : ml_summary,
        "baseline"    : all_summary,
        "ml_lift_pct" : round(ml_lift, 3),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved --> {REPORT_PATH}")

    return ml_trades, all_trades, report