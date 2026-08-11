"""
Per-stock ML engine — unified pipeline.
Merges best of old Nifty pipeline + new per-stock architecture.

Flow: ticker → 10yr OHLCV → features → walk-forward XGBoost → backtest → signal
Target: binary — will tomorrow's Open→Close be positive?
Signal fires after today's market close (3:30 PM IST).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap, os, json
from datetime import date

CACHE_DIR            = "data/stock_cache"
CONFIDENCE_THRESHOLD = 0.60   # must exceed 60% to fire BUY or SELL
BROKERAGE            = 0.001  # 0.1% round-trip

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Features (old pipeline + new) ────────────────────────────────────────────

def build_features(df):
    f = pd.DataFrame(index=df.index)

    # ── From old pipeline ─────────────────────────────────────────────────────
    f['sma20']    = df['Close'].rolling(20).mean()
    f['sma50']    = df['Close'].rolling(50).mean()
    f['sma_ratio']= f['sma20'] / f['sma50']          # > 1 = bullish structure

    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    std_          = df['Close'].rolling(20).std()
    bb_upper      = f['sma20'] + 2 * std_
    bb_lower      = f['sma20'] - 2 * std_
    f['bb_pos']   = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    f['bb_width'] = (bb_upper - bb_lower) / f['sma20']
    f['volatility']= df['Close'].pct_change().rolling(20).std() * 100

    # ── New additions ─────────────────────────────────────────────────────────
    ema12          = df['Close'].ewm(span=12, adjust=False).mean()
    ema26          = df['Close'].ewm(span=26, adjust=False).mean()
    macd           = ema12 - ema26
    f['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f['atr_pct']   = atr / df['Close']               # normalised — comparable across stocks

    vol_avg        = df['Volume'].rolling(20).mean().shift(1)
    f['vol_ratio'] = df['Volume'] / vol_avg
    f['vol_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

    f['ret_1d']    = df['Close'].pct_change(1)  * 100
    f['ret_5d']    = df['Close'].pct_change(5)  * 100
    f['ret_20d']   = df['Close'].pct_change(20) * 100
    f['price_sma20']= df['Close'] / f['sma20']       # how extended from mean

    f['day_of_week']= df.index.dayofweek.astype(float)

    # ── Target: will tomorrow's Open→Close be positive? ───────────────────────
    # Using shift(-1) — same fix as old pipeline's backtest.py
    tomorrow_ret    = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
    f['target']     = (tomorrow_ret > 0).astype(int)   # 1 = green candle, 0 = red

    f.dropna(inplace=True)
    return f, df


FEATURE_COLS = [
    # old pipeline
    'sma_ratio', 'rsi', 'bb_pos', 'bb_width', 'volatility',
    # new
    'macd_hist', 'atr_pct', 'vol_ratio', 'vol_trend',
    'ret_1d', 'ret_5d', 'ret_20d', 'price_sma20', 'day_of_week',
]


# ── Walk-forward train ────────────────────────────────────────────────────────

def walk_forward_train(feat_df):
    X = feat_df[FEATURE_COLS]
    y = feat_df['target']

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    last_model, last_X_test, last_y_test = None, None, None
    last_preds, last_probas = None, None

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
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        roc  = roc_auc_score(y_test, probas[:, 1])

        fold_metrics.append({
            'fold': fold, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'roc_auc': roc,
            'train': len(train_idx), 'test': len(test_idx)
        })

        last_model, last_X_test, last_y_test = model, X_test, y_test
        last_preds, last_probas = preds, probas

    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc    = metrics_df['accuracy'].mean()
    avg_roc    = metrics_df['roc_auc'].mean()

    return last_model, last_X_test, last_y_test, last_preds, last_probas, metrics_df, avg_acc, avg_roc


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(feat_df, raw_df, predictions, probas, X_test):
    res = feat_df.loc[X_test.index].copy()
    res['predicted']  = predictions
    res['conf_buy']   = probas[:, 1]   # P(green candle tomorrow)
    res['conf_sell']  = probas[:, 0]   # P(red candle tomorrow)

    # Next day open→close — using shift(-1) fix from old pipeline
    res['next_open']  = raw_df['Open'].shift(-1).loc[X_test.index]
    res['next_close'] = raw_df['Close'].shift(-1).loc[X_test.index]
    res.dropna(subset=['next_open','next_close'], inplace=True)

    intraday = (res['next_close'] - res['next_open']) / res['next_open']

    # Only trade when confidence exceeds threshold
    buy_mask  = (res['predicted'] == 1) & (res['conf_buy']  >= CONFIDENCE_THRESHOLD)
    sell_mask = (res['predicted'] == 0) & (res['conf_sell'] >= CONFIDENCE_THRESHOLD)

    res['trade_dir']  = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))
    res['ret']        = res['trade_dir'] * intraday - np.where(res['trade_dir'] != 0, BROKERAGE, 0)
    res['equity']     = (1 + res['ret']).cumprod()
    res['market']     = (1 + intraday).cumprod()   # buy-and-hold baseline

    traded = res[res['trade_dir'] != 0]

    if len(traded) == 0:
        return res, {
            'sharpe': 0, 'max_dd': 0, 'win_rate': 0,
            'total_return': 0, 'market_return': round(float((res['market'].iloc[-1]-1)*100),2),
            'trades': 0, 'buy_signals': 0, 'sell_signals': 0
        }

    sharpe   = traded['ret'].mean() / traded['ret'].std() * np.sqrt(252) if traded['ret'].std() > 0 else 0
    max_dd   = (res['equity'] / res['equity'].cummax() - 1).min()
    win_rate = (traded['ret'] > 0).mean()

    return res, {
        'sharpe'       : round(float(sharpe), 3),
        'max_dd'       : round(float(max_dd * 100), 2),
        'win_rate'     : round(float(win_rate * 100), 1),
        'total_return' : round(float((res['equity'].iloc[-1] - 1) * 100), 2),
        'market_return': round(float((res['market'].iloc[-1] - 1) * 100), 2),
        'trades'       : int(len(traded)),
        'buy_signals'  : int((traded['trade_dir'] ==  1).sum()),
        'sell_signals' : int((traded['trade_dir'] == -1).sum()),
    }


# ── Tomorrow's signal ─────────────────────────────────────────────────────────

def get_signal(model, feat_df, raw_df):
    last_row   = feat_df[FEATURE_COLS].iloc[[-1]]
    proba      = model.predict_proba(last_row)[0]
    conf_buy   = float(proba[1])
    conf_sell  = float(proba[0])
    last_close = float(raw_df['Close'].iloc[-1])

    if conf_buy >= CONFIDENCE_THRESHOLD:
        action = 'BUY'
        conf   = conf_buy
    elif conf_sell >= CONFIDENCE_THRESHOLD:
        action = 'SELL'
        conf   = conf_sell
    else:
        action = 'NO ACTION'
        conf   = max(conf_buy, conf_sell)

    # SHAP — robust shape handling
    try:
        explainer = shap.TreeExplainer(model)
        sv_raw    = explainer.shap_values(last_row)
        sv_arr    = np.array(sv_raw)

        if sv_arr.ndim == 3:
            sv = sv_arr[0, :, 1]              # (samples, features, classes)
        elif sv_arr.ndim == 2 and sv_arr.shape[0] == len(FEATURE_COLS):
            sv = sv_arr[:, 1]                 # (features, classes)
        elif sv_arr.ndim == 2:
            sv = sv_arr[1]                    # (classes, features)
        else:
            sv = sv_arr.flatten()[:len(FEATURE_COLS)]

        sv = np.array(sv).flatten()[:len(FEATURE_COLS)]
        drivers = pd.Series(sv, index=FEATURE_COLS).abs().sort_values(ascending=False).head(3)
        top_drivers = [{'feature': k, 'importance': round(float(v), 4)} for k, v in drivers.items()]
    except Exception:
        top_drivers = [{'feature': f, 'importance': 0.0} for f in FEATURE_COLS[:3]]

    return {
        'action'    : action,
        'confidence': round(conf * 100, 1),
        'conf_buy'  : round(conf_buy  * 100, 1),
        'conf_sell' : round(conf_sell * 100, 1),
        'last_close': round(last_close, 2),
        'entry_est' : round(last_close * 1.001, 2),
        'drivers'   : top_drivers,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def analyse(ticker_symbol):
    if not ticker_symbol.endswith('.NS'):
        ticker_symbol += '.NS'
    short      = ticker_symbol.replace('.NS', '')
    cache_path = f"{CACHE_DIR}/{short}_{date.today()}.json"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Download — period='max' so newer stocks still work
    raw = yf.download(ticker_symbol, period='max', interval='1d',
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if len(raw) < 200:
        return {'error': f'Not enough data for {short}. Try the full NSE symbol e.g. KPITTECH, LTIM'}

    raw = raw[['Open','High','Low','Close','Volume']].dropna()

    feat_df, raw_df = build_features(raw)
    model, X_test, y_test, preds, probas, metrics_df, avg_acc, avg_roc = walk_forward_train(feat_df)
    bt_results, bt_metrics = run_backtest(feat_df, raw_df, preds, probas, X_test)
    signal = get_signal(model, feat_df, raw_df)

    folds = []
    for _, row in metrics_df.iterrows():
        folds.append({
            'fold'    : int(row['fold']),
            'accuracy': round(float(row['accuracy']) * 100, 1),
            'roc_auc' : round(float(row['roc_auc']), 3),
            'train'   : int(row['train']),
            'test'    : int(row['test']),
        })

    result = {
        'ticker'      : short,
        'date'        : str(date.today()),
        'rows'        : len(raw),
        'date_from'   : str(raw.index.min().date()),
        'date_to'     : str(raw.index.max().date()),
        'avg_accuracy': round(float(avg_acc) * 100, 1),
        'avg_roc'     : round(float(avg_roc), 3),
        'folds'       : folds,
        'backtest'    : bt_metrics,
        'signal'      : signal,
    }

    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result


if __name__ == '__main__':
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else 'CIPLA'
    print(json.dumps(analyse(t), indent=2))