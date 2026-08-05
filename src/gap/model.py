from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import shap

from .features import FEATURE_COLS

def train_gap_model(feat):
    """
    Walk-forward validation on gap fill dataset.
    Returns last fold model + metrics_df.
    """
    X = feat[FEATURE_COLS]
    y = feat['target']

    tscv = TimeSeriesSplit(n_splits=5)

    fold_metrics = []
    last_model   = None
    last_X_test  = None
    last_y_test  = None
    last_preds   = None

    print("=" * 50)
    print("GAP PREDICTOR — Walk-forward (5 folds)")
    print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        roc  = roc_auc_score(y_test, proba)   # use proba for ROC, more informative

        fold_metrics.append({
            'fold': fold, 'train_size': len(train_idx), 'test_size': len(test_idx),
            'accuracy': acc, 'precision': prec, 'recall': rec, 'roc_auc': roc,
        })

        print(f"\nFold {fold}  |  train={len(train_idx)}  test={len(test_idx)}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  ROC-AUC  : {roc:.4f}  ← use this as primary metric")

        last_model  = model
        last_X_test = X_test
        last_y_test = y_test
        last_preds  = preds

    metrics_df = pd.DataFrame(fold_metrics)

    print("\n" + "=" * 50)
    print("AVERAGE ACROSS FOLDS")
    print("=" * 50)
    print(f"  Accuracy : {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
    print(f"  Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  Recall   : {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print(f"  ROC-AUC  : {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")

    # SHAP
    print("\nSHAP FEATURE IMPORTANCE (last fold)")
    explainer   = shap.TreeExplainer(last_model)
    shap_values = explainer.shap_values(last_X_test)
    shap_imp    = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(shap_imp)

    return last_model, last_preds, last_X_test, last_y_test, metrics_df