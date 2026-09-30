"""
XGBoost RANKING model for earnings drift.

Key design decisions:
  - Regression not classification: predict 10d return magnitude
  - Walk-forward CV with TimeSeriesSplit (no future leakage)
  - Primary metric = lift (top 25% trades vs all trades)
  - ALL fold predictions collected → full_pred_df for backtest
  - Model saved to models/ so dashboard can load it without retraining
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import shap, os, json
from .features import FEATURE_COLS

MODEL_PATH   = "models/earnings_drift.json"
METRICS_PATH = "models/earnings_metrics.json"


def train_model(feat_df, n_splits=5):
    feat_df = feat_df.sort_index()
    X = feat_df[FEATURE_COLS]
    y = feat_df["target_10d"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    all_pred_rows = []          # ← collect every fold's predictions
    last_model = last_X_test = last_test_df = None

    print("=" * 58)
    print("EARNINGS DRIFT — Walk-forward XGBoost Regressor")
    print("=" * 58)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators     = 400,
            max_depth        = 3,
            learning_rate    = 0.02,
            subsample        = 0.75,
            colsample_bytree = 0.75,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        test_result              = feat_df.iloc[test_idx].copy()
        test_result["predicted"] = preds
        top_q = test_result[test_result["predicted"] >= np.percentile(preds, 75)]

        top_mean = float(top_q["target_10d"].mean())
        all_mean = float(test_result["target_10d"].mean())
        lift     = top_mean - all_mean

        fold_metrics.append({
            "fold": fold, "train": len(train_idx), "test": len(test_idx),
            "mae": round(mae, 3), "r2": round(r2, 4),
            "top25_return": round(top_mean, 3),
            "all_return"  : round(all_mean, 3),
            "lift"        : round(lift, 3),
        })

        print(f"\nFold {fold}  train={len(train_idx)}  test={len(test_idx)}")
        print(f"  MAE         : {mae:.2f}%")
        print(f"  R²          : {r2:.4f}")
        print(f"  All trades  : {all_mean:.2f}%  avg 10d return")
        print(f"  Top 25%     : {top_mean:.2f}%  ← trades we take")
        print(f"  Lift        : {lift:+.2f}%")

        # collect this fold's predictions into the running list
        all_pred_rows.append(test_result)

        last_model   = model
        last_X_test  = X_test
        last_test_df = test_result

    # Full prediction dataset across all folds — used for the proper backtest
    full_pred_df = pd.concat(all_pred_rows).sort_index()

    metrics_df = pd.DataFrame(fold_metrics)
    avg_lift   = metrics_df["lift"].mean()

    print("\n" + "=" * 58)
    print("SUMMARY")
    print(f"  Avg lift (top25 vs all) : {avg_lift:+.2f}%")
    print(f"  Avg top25 return        : {metrics_df['top25_return'].mean():.2f}%")
    print(f"  Avg all-trades return   : {metrics_df['all_return'].mean():.2f}%")
    print(f"  Total predicted events  : {len(full_pred_df)}")

    # SHAP feature importance
    print("\nSHAP FEATURE IMPORTANCE (last fold)")
    explainer   = shap.TreeExplainer(last_model)
    shap_values = explainer.shap_values(last_X_test)
    shap_imp    = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(shap_imp.to_string())

    # Save model + metrics
    os.makedirs("models", exist_ok=True)
    last_model.save_model(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    summary = {
        "avg_lift"        : round(avg_lift, 3),
        "avg_top25_return": round(metrics_df["top25_return"].mean(), 3),
        "avg_all_return"  : round(metrics_df["all_return"].mean(), 3),
        "n_events"        : len(feat_df),
        "n_stocks"        : int(feat_df["ticker"].nunique()),
        "date_from"       : str(feat_df.index.min().date()),
        "date_to"         : str(feat_df.index.max().date()),
        "folds"           : fold_metrics,
        "shap_importance" : shap_imp.round(4).to_dict(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    return last_model, last_test_df, metrics_df, summary, full_pred_df


def load_model():
    """Load saved model for dashboard use."""
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model(MODEL_PATH)
    return m