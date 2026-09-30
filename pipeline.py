"""
NiftyML — full training pipeline.
Run this once to train the model. Dashboard loads the saved model.

Usage:
    python pipeline.py           # use cached data
    python pipeline.py --fresh   # force re-download everything
"""

import sys, os

FRESH = "--fresh" in sys.argv

print("\n" + "=" * 58)
print("  NiftyML — Earnings Drift Pipeline")
print("=" * 58)

# Step 1 — Download
print("\n[1/5] Downloading OHLCV data...")
from src.earnings.data_loader import download_all, build_events
download_all(force=FRESH)

# Step 2 — Detect events
print("\n[2/5] Detecting earnings gap events...")
events = build_events(force=FRESH)
if events.empty:
    print("ERROR: No events found. Check network and data.")
    sys.exit(1)

# Step 3 — Build features
print("\n[3/5] Building features...")
from src.earnings.features import build_features
feat_df = build_features(events)
if len(feat_df) < 50:
    print(f"WARNING: Only {len(feat_df)} events with complete features. Model may be weak.")

# Step 4 — Train model
print("\n[4/5] Training model...")
from src.earnings.model import train_model
model, test_df, metrics_df, summary, full_pred_df = train_model(feat_df)

# Step 5 — Backtest on full prediction dataset
print("\n[5/5] Running backtest...")
from src.earnings.backtest import run_backtest
ml_trades, all_trades, report = run_backtest(test_df, full_feat_df=full_pred_df)

print("\n" + "=" * 58)
print("  PIPELINE COMPLETE")
print(f"  Model       → models/earnings_drift.json")
print(f"  Report      → reports/backtest_summary.json")
print(f"  Events      : {len(feat_df)} across {feat_df['ticker'].nunique()} stocks")
print(f"  ML lift     : {report['ml_lift_pct']:+.3f}% per trade vs baseline")
print(f"  ML win rate : {report['ml_strategy'].get('win_rate', 0)}%")
print(f"  ML Sharpe   : {report['ml_strategy'].get('sharpe', 0)}")
print("=" * 58)