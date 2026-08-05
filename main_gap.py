from src.gap.data_loader import load_gap_data
from src.gap.features import build_gap_features
from src.gap.model import train_gap_model
from src.gap.backtest import run_gap_backtest

# 1. Load
raw = load_gap_data()

# 2. Features
feat = build_gap_features(raw)

# 3. Train + walk-forward
model, predictions, X_test, y_test, metrics_df = train_gap_model(feat)

# 4. Backtest
results, sharpe, max_dd, win_rate = run_gap_backtest(feat, predictions, X_test)