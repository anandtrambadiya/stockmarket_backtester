"""
ML model for earnings drift.
Task: regression — predict 15-day forward return magnitude.
We use the prediction to RANK trades, not as absolute targets.
Top quartile predictions → take the trade.
"""

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import shap
from .features import FEATURE_COLS

def train_earnings_model(feat_df):
    """
    Walk-forward validation.
    Predict 15-day return. Use predictions to rank trades.
    Returns last fold model + predictions on last fold test set.
    """
    feat_df = feat_df.sort_index()
    X = feat_df[FEATURE_COLS]
    y = feat_df['target_15d']

    tscv = TimeSeriesSplit(n_splits=5)

    fold_metrics = []
    last_model   = None
    last_X_test  = None
    last_test_df = None

    print("=" * 55)
    print("EARNINGS DRIFT MODEL — Walk-forward (5 folds)")
    print("=" * 55)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        # rank-based metric — does top quartile of predictions actually perform better?
        test_result = feat_df.iloc[test_idx].copy()
        test_result['predicted'] = preds
        top_q  = test_result[test_result['predicted'] >= np.percentile(preds, 75)]
        all_q  = test_result

        top_mean = top_q['target_15d'].mean()
        all_mean = all_q['target_15d'].mean()

        fold_metrics.append({
            'fold': fold,
            'train': len(train_idx), 'test': len(test_idx),
            'mae': mae, 'r2': r2,
            'top_q_return': top_mean,
            'all_return': all_mean,
            'lift': top_mean - all_mean,
        })

        print(f"\nFold {fold}  |  train={len(train_idx)}  test={len(test_idx)}")
        print(f"  MAE            : {mae:.2f}%")
        print(f"  R²             : {r2:.4f}")
        print(f"  All trades avg : {all_mean:.2f}%  (15d return)")
        print(f"  Top 25% avg    : {top_mean:.2f}%  ← trades we'd actually take")
        print(f"  Lift           : {top_mean - all_mean:+.2f}%")

        last_model   = model
        last_X_test  = X_test
        last_test_df = test_result

    metrics_df = pd.DataFrame(fold_metrics)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Avg lift (top 25% vs all) : {metrics_df['lift'].mean():+.2f}%")
    print(f"  Avg top 25% return        : {metrics_df['top_q_return'].mean():.2f}%")
    print(f"  Avg all-trades return     : {metrics_df['all_return'].mean():.2f}%")

    # SHAP
    print("\nSHAP FEATURE IMPORTANCE (last fold)")
    explainer   = shap.TreeExplainer(last_model)
    shap_values = explainer.shap_values(last_X_test)
    shap_imp    = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(shap_imp)

    return last_model, last_test_df, metrics_df