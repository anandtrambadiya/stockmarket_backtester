from src.earnings.data_loader import download_all, build_events
from src.earnings.features import build_features
from src.earnings.model import train_earnings_model
from src.earnings.backtest import run_earnings_backtest

# Step 1 — Download (skips cached tickers)
print("=" * 55)
print("STEP 1: Data download")
print("=" * 55)
download_all()

# Step 2 — Detect results days
print("\nSTEP 2: Detecting results days...")
events = build_events()

# Step 3 — Build features
print("\nSTEP 3: Building features...")
feat_df = build_features(events)

# Step 4 — Train model
print("\nSTEP 4: Training model...")
model, test_df, metrics_df = train_earnings_model(feat_df)

# Step 5 — Backtest
print("\nSTEP 5: Backtesting...")
trades, baseline = run_earnings_backtest(test_df)