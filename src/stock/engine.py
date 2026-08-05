"""
Per-stock ML engine.
Pull 10yr data → build features → walk-forward train → backtest → tomorrow's signal.
Single entry point: analyse(ticker) returns everything needed for the dashboard.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap
import os, json
from datetime import date, timedelta

CACHE_DIR     = "data/stock_cache"
CONFIDENCE_THRESHOLD = 0.62   # model must be > 62% confident to say BUY or SELL
                               # below this → NO ACTION

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Features ──────────────────────────────────────────────────────────────────

def build_features(df):
    f = pd.DataFrame(index=df.index)

    # Price-based
    f['sma20']       = df['Close'].rolling(20).mean()
    f['sma50']       = df['Close'].rolling(50).mean()
    f['sma200']      = df['Close'].rolling(200).mean()
    f['sma_ratio']   = f['sma20'] / f['sma50']           # above 1 = bullish
    f['price_sma20'] = df['Close'] / f['sma20']          # how extended from 20d mean

    # RSI-14
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # Bollinger Bands
    std_          = df['Close'].rolling(20).std()
    bb_upper      = f['sma20'] + 2 * std_
    bb_lower      = f['sma20'] - 2 * std_
    f['bb_pos']   = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 0=at lower, 1=at upper
    f['bb_width'] = (bb_upper - bb_lower) / f['sma20']

    # MACD
    ema12         = df['Close'].ewm(span=12, adjust=False).mean()
    ema26         = df['Close'].ewm(span=26, adjust=False).mean()
    macd          = ema12 - ema26
    macd_signal   = macd.ewm(span=9, adjust=False).mean()
    f['macd_hist']= macd - macd_signal     # positive = bullish momentum

    # ATR-14 — volatility regime
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    f['atr']     = tr.rolling(14).mean()
    f['atr_pct'] = f['atr'] / df['Close']   # normalised ATR

    # Volume features
    f['vol_ratio']  = df['Volume'] / df['Volume'].rolling(20).mean().shift(1)
    f['vol_trend']  = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

    # Return features (momentum)
    f['ret_1d']  = df['Close'].pct_change(1)  * 100
    f['ret_5d']  = df['Close'].pct_change(5)  * 100
    f['ret_20d'] = df['Close'].pct_change(20) * 100

    # Calendar
    f['day_of_week'] = df.index.dayofweek.astype(float)
    f['month']       = df.index.month.astype(float)

    # ── Target: will tomorrow's candle be profitable? ──────────────────────────
    # BUY signal  = tomorrow Open→Close up > 0.3%  → label 2
    # SELL signal = tomorrow Open→Close down > 0.3% → label 0
    # NO ACTION   = move < 0.3% either way           → label 1
    tomorrow_ret = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1) * 100
    f['target'] = np.where(tomorrow_ret >  0.3, 2,
                  np.where(tomorrow_ret < -0.3, 0, 1))

    f.dropna(inplace=True)
    f['target'] = f['target'].astype(int)
    return f, df


FEATURE_COLS = [
    'sma_ratio','price_sma20','rsi','bb_pos','bb_width',
    'macd_hist','atr_pct','vol_ratio','vol_trend',
    'ret_1d','ret_5d','ret_20d','day_of_week','month'
]


# ── Walk-forward train ─────────────────────────────────────────────────────────

def walk_forward_train(feat_df):
    X = feat_df[FEATURE_COLS]
    y = feat_df['target']

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    last_model   = None
    last_X_test  = None
    last_y_test  = None
    last_preds   = None
    last_probas  = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            num_class=3,
            objective='multi:softprob',
        )
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, preds)
        fold_metrics.append({'fold': fold, 'accuracy': acc,
                             'train': len(train_idx), 'test': len(test_idx)})

        last_model  = model
        last_X_test = X_test
        last_y_test = y_test
        last_preds  = preds
        last_probas = probas

    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc    = metrics_df['accuracy'].mean()

    return last_model, last_X_test, last_y_test, last_preds, last_probas, metrics_df, avg_acc


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(feat_df, raw_df, predictions, probas, X_test):
    results = feat_df.loc[X_test.index].copy()
    results['predicted'] = predictions
    results['conf_buy']  = probas[:, 2]   # confidence for BUY
    results['conf_sell'] = probas[:, 0]   # confidence for SELL

    # Shift to get next day open/close
    results['next_open']  = raw_df['Open'].shift(-1).loc[X_test.index]
    results['next_close'] = raw_df['Close'].shift(-1).loc[X_test.index]
    results.dropna(subset=['next_open','next_close'], inplace=True)

    intraday = (results['next_close'] - results['next_open']) / results['next_open']

    # Only act when confidence exceeds threshold
    buy_mask  = (results['predicted'] == 2) & (results['conf_buy']  > CONFIDENCE_THRESHOLD)
    sell_mask = (results['predicted'] == 0) & (results['conf_sell'] > CONFIDENCE_THRESHOLD)

    results['trade_dir'] = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))
    results['ret']       = results['trade_dir'] * intraday - np.where(results['trade_dir'] != 0, 0.001, 0)

    # Cumulative — compounding only on trade days
    results['equity'] = (1 + results['ret']).cumprod()
    results['market'] = (1 + intraday).cumprod()

    traded = results[results['trade_dir'] != 0]
    if len(traded) == 0:
        return results, {'sharpe': 0, 'max_dd': 0, 'win_rate': 0,
                         'total_return': 0, 'market_return': 0,
                         'trades': 0, 'buy_signals': 0, 'sell_signals': 0}

    ann     = np.sqrt(252)
    sharpe  = traded['ret'].mean() / traded['ret'].std() * ann if traded['ret'].std() > 0 else 0
    max_dd  = (results['equity'] / results['equity'].cummax() - 1).min()
    win_rate= (traded['ret'] > 0).mean()
    tot_ret = results['equity'].iloc[-1] - 1
    mkt_ret = results['market'].iloc[-1] - 1

    return results, {
        'sharpe'       : round(float(sharpe), 3),
        'max_dd'       : round(float(max_dd * 100), 2),
        'win_rate'     : round(float(win_rate * 100), 1),
        'total_return' : round(float(tot_ret * 100), 2),
        'market_return': round(float(mkt_ret * 100), 2),
        'trades'       : int(len(traded)),
        'buy_signals'  : int((traded['trade_dir'] == 1).sum()),
        'sell_signals' : int((traded['trade_dir'] == -1).sum()),
    }


# ── Tomorrow's signal ─────────────────────────────────────────────────────────

def get_signal(model, feat_df, raw_df):
    last_row    = feat_df[FEATURE_COLS].iloc[[-1]]
    proba       = model.predict_proba(last_row)[0]
    pred        = model.predict(last_row)[0]

    conf_buy    = float(proba[2])
    conf_sell   = float(proba[0])
    conf_hold   = float(proba[1])

    if pred == 2 and conf_buy > CONFIDENCE_THRESHOLD:
        action = 'BUY'
        conf   = conf_buy
    elif pred == 0 and conf_sell > CONFIDENCE_THRESHOLD:
        action = 'SELL'
        conf   = conf_sell
    else:
        action = 'NO ACTION'
        conf   = conf_hold

    last_close = float(raw_df['Close'].iloc[-1])

    # SHAP explanation — top 3 drivers
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(last_row)

    # XGBoost multiclass SHAP can return:
    # - list of arrays, one per class: each shape (1, n_features)
    # - single 3D array: (1, n_features, n_classes)
    # - single 2D array: (n_features, n_classes)  ← what we're getting
    sv_arr = np.array(shap_vals)
    if sv_arr.ndim == 3:
        sv = sv_arr[0, :, 2]          # (samples, features, classes) → BUY class
    elif sv_arr.ndim == 2:
        if sv_arr.shape[0] == len(FEATURE_COLS):
            sv = sv_arr[:, 2]          # (features, classes) → BUY class
        else:
            sv = sv_arr[0]             # (samples, features)
    elif isinstance(shap_vals, list):
        sv = np.array(shap_vals[2])[0] # list[class][sample]
    else:
        sv = sv_arr.flatten()[:len(FEATURE_COLS)]

    sv = np.array(sv).flatten()[:len(FEATURE_COLS)]
    drivers = pd.Series(sv, index=FEATURE_COLS).abs().sort_values(ascending=False).head(3)
    top_drivers = [{'feature': k, 'importance': round(float(v), 4)}
                   for k, v in drivers.items()]

    return {
        'action'     : action,
        'confidence' : round(conf * 100, 1),
        'conf_buy'   : round(conf_buy  * 100, 1),
        'conf_sell'  : round(conf_sell * 100, 1),
        'conf_hold'  : round(conf_hold * 100, 1),
        'last_close' : round(last_close, 2),
        'entry_est'  : round(last_close * 1.001, 2),   # approx next open (0.1% gap estimate)
        'drivers'    : top_drivers,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def analyse(ticker_symbol):
    """
    Full pipeline for one stock.
    ticker_symbol: e.g. 'KPIT.NS' or just 'KPIT' (we add .NS if missing)
    Returns dict with everything the dashboard needs.
    """
    if not ticker_symbol.endswith('.NS'):
        ticker_symbol += '.NS'

    short = ticker_symbol.replace('.NS','')
    cache_path = f"{CACHE_DIR}/{short}_{date.today()}.json"

    # Daily cache
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Try with .NS suffix, fall back to TECHM style if needed
    raw = yf.download(ticker_symbol, period='max', interval='1d',
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # If .NS fails, yfinance sometimes needs different suffix — try as-is
    if len(raw) < 50:
        alt = ticker_symbol.replace('.NS', '-NS')
        raw = yf.download(alt, period='max', interval='1d',
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    if len(raw) < 200:
        return {'error': f'Not enough data for {short} — try the full NSE ticker (e.g. KPITTECH.NS, LTIM.NS)'}

    raw = raw[['Open','High','Low','Close','Volume']].dropna()

    # Features
    feat_df, raw_df = build_features(raw)

    # Train
    model, X_test, y_test, preds, probas, metrics_df, avg_acc = walk_forward_train(feat_df)

    # Backtest
    bt_results, bt_metrics = run_backtest(feat_df, raw_df, preds, probas, X_test)

    # Signal
    signal = get_signal(model, feat_df, raw_df)

    # Fold summary for display
    folds = metrics_df[['fold','accuracy','train','test']].round(4).to_dict('records')

    result = {
        'ticker'      : short,
        'date'        : str(date.today()),
        'rows'        : len(raw),
        'date_from'   : str(raw.index.min().date()),
        'date_to'     : str(raw.index.max().date()),
        'avg_accuracy': round(float(avg_acc * 100), 1),
        'folds'       : folds,
        'backtest'    : bt_metrics,
        'signal'      : signal,
    }

    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result


if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'KPIT'
    result = analyse(ticker)
    print(json.dumps(result, indent=2))