"""
Per-stock ML engine — daily signal analyser.

Pipeline:
  1. Download max history via yfinance
  2. Build technical features (SMA, EMA, RSI, BB, MACD, ATR, volume, momentum)
  3. Target = will this stock outperform Nifty500 over next 5 days?
     (relative return removes market noise — much more learnable than raw direction)
  4. XGBoost Classifier (BUY=2 / HOLD=1 / SELL=0) with 62% confidence threshold
  5. Walk-forward CV — 5 folds, no future leakage
  6. Backtest: signal fires after close, trade next open→close, measure real P&L
  7. SHAP top-3 drivers for today's signal
  8. Learning summary: accuracy, Sharpe, win rate, equity curve vs market
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import shap, os, json, warnings
from datetime import date

warnings.filterwarnings('ignore')

CACHE_DIR            = "data/stock_cache"
CONFIDENCE_THRESHOLD = 0.62    # model must be >62% confident to act
BROKERAGE            = 0.001   # 0.1% round-trip
MIN_ROWS             = 400     # need at least 400 days (~1.6 years) to train

os.makedirs(CACHE_DIR, exist_ok=True)

# Indices that cannot be traded — return helpful error
INDEX_KEYWORDS = {'NIFTY', 'SENSEX', 'BANKNIFTY', 'FINNIFTY', 'MIDCAP',
                  'SMALLCAP', 'VIX', 'BSE', 'NSE'}

FEATURE_COLS = [
    'sma_ratio',      # SMA20/SMA50 — trend direction
    'ema_ratio',      # EMA12/EMA26 — short vs long momentum
    'price_sma20',    # close/SMA20 — how extended from mean
    'rsi',            # RSI-14
    'bb_pos',         # position within Bollinger Bands (0=lower, 1=upper)
    'bb_width',       # Bollinger Band width — volatility regime
    'macd_hist',      # MACD histogram — momentum shift
    'atr_pct',        # ATR/close — normalised daily volatility
    'vol_ratio',      # today's volume / 20d avg
    'vol_trend',      # 5d avg volume / 20d avg volume
    'ret_1d',         # 1-day return
    'ret_5d',         # 5-day return
    'ret_20d',        # 20-day return
    'rel_ret_5d',     # 5d return relative to Nifty500 (stock - market)
    'day_of_week',    # 0=Mon, 3=Thu (F&O expiry), 4=Fri
    'month',          # month of year (results season patterns)
]


# ── Ticker resolution ─────────────────────────────────────────────────────────

def resolve_ticker(user_input):
    """
    Takes what the user typed, returns (yfinance_symbol, display_name, error).
    Handles: plain names, .NS suffix, indices, common aliases.
    """
    t = user_input.strip().upper().replace(' ', '')

    # Index check — can't trade these
    for kw in INDEX_KEYWORDS:
        if kw in t and not any(x in t for x in ['BEES', 'ETF', 'FUND']):
            return None, t, (
                f"'{t}' looks like an index — indices can't be traded directly. "
                f"Try NIFTYBEES.NS (Nifty ETF) or a specific stock like RELIANCE, HDFCBANK, COFORGE."
            )

    # Already has suffix
    if t.endswith('.NS') or t.endswith('.BO'):
        return t, t.replace('.NS','').replace('.BO',''), None

    # Common aliases
    aliases = {
        'KPIT':      'KPITTECH.NS',
        'L&T':       'LT.NS',
        'LNT':       'LT.NS',
        'M&M':       'M&M.NS',
        'MM':        'M&M.NS',
        'BAJAJAUTO': 'BAJAJ-AUTO.NS',
        'BAJAJFINSERV': 'BAJAJFINSV.NS',
        'HDFCLIFE':  'HDFCLIFE.NS',
        'SBILIFE':   'SBILIFE.NS',
        'LTMINDTREE':'LTIM.NS',
        'LTIM':      'LTIM.NS',
        'MINDTREE':  'LTIM.NS',
        'WIPRO':     'WIPRO.NS',
        'INFY':      'INFY.NS',
        'TCS':       'TCS.NS',
    }
    if t in aliases:
        sym = aliases[t]
        return sym, sym.replace('.NS',''), None

    # Default: add .NS
    return f"{t}.NS", t, None


# ── Data download ─────────────────────────────────────────────────────────────

def download_stock(symbol):
    df = yf.download(symbol, period='max', interval='1d',
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) < 10:
        return None
    return df[['Open','High','Low','Close','Volume']].dropna()


def download_nifty500():
    """Download Nifty 500 for relative return calculation."""
    try:
        df = yf.download('^CRSLDX', period='max', interval='1d',
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df['Close'].dropna()
    except Exception:
        return None


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df, nifty_close=None):
    f = pd.DataFrame(index=df.index)

    # Moving averages
    f['sma20']       = df['Close'].rolling(20).mean()
    f['sma50']       = df['Close'].rolling(50).mean()
    f['ema12']       = df['Close'].ewm(span=12, adjust=False).mean()
    f['ema26']       = df['Close'].ewm(span=26, adjust=False).mean()
    f['sma_ratio']   = f['sma20'] / f['sma50']
    f['ema_ratio']   = f['ema12'] / f['ema26']
    f['price_sma20'] = df['Close'] / f['sma20']

    # RSI-14
    delta  = df['Close'].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # Bollinger Bands (20, 2)
    std_         = df['Close'].rolling(20).std()
    bb_upper     = f['sma20'] + 2 * std_
    bb_lower     = f['sma20'] - 2 * std_
    f['bb_pos']  = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    f['bb_width']= (bb_upper - bb_lower) / f['sma20']

    # MACD
    f['macd_hist'] = (f['ema12'] - f['ema26']) - \
                     (f['ema12'] - f['ema26']).ewm(span=9, adjust=False).mean()

    # ATR-14
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    f['atr']     = tr.rolling(14).mean()
    f['atr_pct'] = f['atr'] / df['Close']

    # Volume
    vol_avg      = df['Volume'].rolling(20).mean().shift(1)
    f['vol_ratio']= df['Volume'] / vol_avg
    f['vol_trend']= df['Volume'].rolling(5).mean() / \
                    df['Volume'].rolling(20).mean()

    # Momentum
    f['ret_1d']  = df['Close'].pct_change(1)  * 100
    f['ret_5d']  = df['Close'].pct_change(5)  * 100
    f['ret_20d'] = df['Close'].pct_change(20) * 100

    # Relative return vs Nifty500 (key feature — removes market noise)
    if nifty_close is not None:
        nifty_aligned = nifty_close.reindex(df.index, method='ffill')
        nifty_ret_5d  = nifty_aligned.pct_change(5) * 100
        f['rel_ret_5d'] = f['ret_5d'] - nifty_ret_5d
    else:
        f['rel_ret_5d'] = f['ret_5d']  # fallback: use absolute return

    # Calendar
    f['day_of_week'] = df.index.dayofweek.astype(float)
    f['month']       = df.index.month.astype(float)

    # ── Target: relative outperformance over next 5 days ──────────────────────
    # Will this stock beat Nifty500 by >0.5% over next 5 days? → BUY=2
    # Will it underperform by >0.5%? → SELL=0
    # Otherwise → HOLD=1
    stock_5d_fwd = df['Close'].pct_change(5).shift(-5) * 100
    if nifty_close is not None:
        nifty_5d_fwd = nifty_aligned.pct_change(5).shift(-5) * 100
        rel_fwd      = stock_5d_fwd - nifty_5d_fwd
    else:
        rel_fwd = stock_5d_fwd

    f['target'] = np.where(rel_fwd >  0.5, 2,
                  np.where(rel_fwd < -0.5, 0, 1))

    f.dropna(inplace=True)
    f['target'] = f['target'].astype(int)

    # Drop helper columns
    f.drop(columns=['sma20','sma50','ema12','ema26','atr'], inplace=True)

    return f


# ── Walk-forward training ─────────────────────────────────────────────────────

def walk_forward_train(feat_df):
    X = feat_df[FEATURE_COLS]
    y = feat_df['target']

    n_splits = min(5, len(feat_df) // 80)
    if n_splits < 2:
        return None, None, None, None, None, pd.DataFrame(), 0.0

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    last_model = last_X_test = last_y_test = last_preds = last_probas = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.03,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            eval_metric      = 'mlogloss',
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
            num_class        = 3,
            objective        = 'multi:softprob',
        )
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)
        acc    = accuracy_score(y_test, preds)

        # Distribution in test set
        buy_pct  = int((y_test == 2).mean() * 100)
        hold_pct = int((y_test == 1).mean() * 100)
        sell_pct = int((y_test == 0).mean() * 100)

        fold_metrics.append({
            'fold': fold, 'accuracy': round(float(acc) * 100, 1),
            'train': len(train_idx), 'test': len(test_idx),
            'buy_pct': buy_pct, 'hold_pct': hold_pct, 'sell_pct': sell_pct,
        })
        last_model  = model
        last_X_test = X_test
        last_y_test = y_test
        last_preds  = preds
        last_probas = probas

    metrics_df = pd.DataFrame(fold_metrics)
    avg_acc    = float(metrics_df['accuracy'].mean())
    return last_model, last_X_test, last_y_test, last_preds, last_probas, metrics_df, avg_acc


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(feat_df, raw_df, predictions, probas, X_test):
    results = feat_df.loc[X_test.index].copy()
    results['predicted'] = predictions
    results['conf_buy']  = probas[:, 2]
    results['conf_sell'] = probas[:, 0]

    # Next day OHLC
    results['next_open']  = raw_df['Open'].shift(-1).loc[X_test.index]
    results['next_close'] = raw_df['Close'].shift(-1).loc[X_test.index]
    results.dropna(subset=['next_open', 'next_close'], inplace=True)

    intraday = (results['next_close'] - results['next_open']) / results['next_open']

    # Only trade when confidence > threshold
    buy_mask  = (results['predicted'] == 2) & (results['conf_buy']  > CONFIDENCE_THRESHOLD)
    sell_mask = (results['predicted'] == 0) & (results['conf_sell'] > CONFIDENCE_THRESHOLD)

    results['trade_dir'] = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))
    results['ret']       = results['trade_dir'] * intraday - \
                           np.where(results['trade_dir'] != 0, BROKERAGE, 0)
    results['equity']    = (1 + results['ret']).cumprod()
    results['market']    = (1 + intraday).cumprod()

    traded = results[results['trade_dir'] != 0]

    if len(traded) == 0:
        return results, {
            'sharpe': 0, 'max_dd': 0, 'win_rate': 0,
            'total_return': 0, 'market_return': 0,
            'trades': 0, 'buy_signals': 0, 'sell_signals': 0,
        }

    sharpe   = traded['ret'].mean() / traded['ret'].std() * np.sqrt(252) \
               if traded['ret'].std() > 0 else 0
    max_dd   = float((results['equity'] / results['equity'].cummax() - 1).min() * 100)
    win_rate = float((traded['ret'] > 0).mean() * 100)
    tot_ret  = float((results['equity'].iloc[-1] - 1) * 100)
    mkt_ret  = float((results['market'].iloc[-1] - 1) * 100)

    # Daily equity curve (last 60 points for display)
    equity_curve = (results['equity'].iloc[-60:] - 1) * 100
    market_curve = (results['market'].iloc[-60:] - 1) * 100

    return results, {
        'sharpe'       : round(float(sharpe), 3),
        'max_dd'       : round(max_dd, 2),
        'win_rate'     : round(win_rate, 1),
        'total_return' : round(tot_ret, 2),
        'market_return': round(mkt_ret, 2),
        'trades'       : int(len(traded)),
        'buy_signals'  : int((traded['trade_dir'] == 1).sum()),
        'sell_signals' : int((traded['trade_dir'] == -1).sum()),
        'equity_curve' : [round(x, 2) for x in equity_curve.tolist()],
        'market_curve' : [round(x, 2) for x in market_curve.tolist()],
    }


# ── Today's signal ────────────────────────────────────────────────────────────

def get_signal(model, feat_df, raw_df):
    if model is None or len(feat_df) == 0:
        return {'action': 'NO DATA', 'confidence': 0, 'drivers': []}

    last_row   = feat_df[FEATURE_COLS].iloc[[-1]]
    proba      = model.predict_proba(last_row)[0]
    pred       = int(model.predict(last_row)[0])

    conf_buy   = float(proba[2])
    conf_sell  = float(proba[0])
    conf_hold  = float(proba[1])

    if pred == 2 and conf_buy > CONFIDENCE_THRESHOLD:
        action, conf = 'BUY', conf_buy
    elif pred == 0 and conf_sell > CONFIDENCE_THRESHOLD:
        action, conf = 'SELL', conf_sell
    else:
        action, conf = 'NO ACTION', conf_hold

    last_close = float(raw_df['Close'].iloc[-1])

    # SHAP — top 3 feature drivers
    try:
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(last_row)
        sv_arr    = np.array(sv)

        if sv_arr.ndim == 3:
            sv_buy = sv_arr[0, :, 2]
        elif sv_arr.ndim == 2 and sv_arr.shape[0] == len(FEATURE_COLS):
            sv_buy = sv_arr[:, 2]
        elif isinstance(sv, list):
            sv_buy = np.array(sv[2])[0]
        else:
            sv_buy = sv_arr.flatten()[:len(FEATURE_COLS)]

        sv_buy = np.array(sv_buy).flatten()[:len(FEATURE_COLS)]
        drivers = pd.Series(sv_buy, index=FEATURE_COLS).abs() \
                    .sort_values(ascending=False).head(3)
        top_drivers = [{'feature': k, 'importance': round(float(v), 4),
                        'direction': 'bullish' if sv_buy[FEATURE_COLS.index(k)] > 0 else 'bearish'}
                       for k, v in drivers.items()]
    except Exception:
        top_drivers = []

    return {
        'action'     : action,
        'confidence' : round(conf * 100, 1),
        'conf_buy'   : round(conf_buy  * 100, 1),
        'conf_sell'  : round(conf_sell * 100, 1),
        'conf_hold'  : round(conf_hold * 100, 1),
        'last_close' : round(last_close, 2),
        'drivers'    : top_drivers,
    }


# ── Technical snapshot ────────────────────────────────────────────────────────

def technical_snapshot(df):
    close  = df['Close']
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

    ema12     = close.ewm(span=12).mean()
    ema26     = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9).mean()
    macd_hist = float((macd_line - macd_sig).iloc[-1])

    vol20 = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    ret1m = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
    ret3m = float((close.iloc[-1] / close.iloc[-63] - 1) * 100)
    ret1y = float((close.iloc[-1] / close.iloc[-252] - 1) * 100) \
            if len(close) > 252 else None

    cur   = float(close.iloc[-1])
    trend = 'Bullish' if sma20 > sma50 else 'Bearish'
    above200 = cur > sma200

    # Volume surge today
    vol_avg = float(df['Volume'].rolling(20).mean().iloc[-1])
    vol_now = float(df['Volume'].iloc[-1])
    vol_ratio = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0

    return {
        'last_close' : round(cur, 2),
        'sma20'      : round(sma20, 2),
        'sma50'      : round(sma50, 2),
        'rsi'        : round(rsi, 1),
        'macd_hist'  : round(macd_hist, 4),
        'vol_ann_pct': round(vol20, 1),
        'ret_1m'     : round(ret1m, 2),
        'ret_3m'     : round(ret3m, 2),
        'ret_1y'     : round(ret1y, 2) if ret1y is not None else None,
        'sma_trend'  : trend,
        'above_sma200': above200,
        'vol_ratio'  : vol_ratio,
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def analyse(user_input):
    symbol, short, err = resolve_ticker(user_input)

    if err:
        return {'error': err, 'ticker': user_input}

    cache_path = f"{CACHE_DIR}/{short}_{date.today()}.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Download
    raw = download_stock(symbol)
    if raw is None or len(raw) < MIN_ROWS:
        # Try .BO (BSE) as fallback
        raw2 = download_stock(symbol.replace('.NS', '.BO'))
        if raw2 is not None and len(raw2) >= MIN_ROWS:
            raw = raw2
        else:
            return {
                'error': f"Not enough data for '{short}'. "
                         f"Need {MIN_ROWS}+ trading days. "
                         f"Check the NSE symbol — try e.g. KPITTECH not KPIT, LT not L&T.",
                'ticker': short,
            }

    nifty = download_nifty500()
    tech  = technical_snapshot(raw)

    # Features + target
    feat_df = build_features(raw, nifty)
    if len(feat_df) < 200:
        return {'error': f'Not enough feature data for {short} after rolling windows.', 'ticker': short}

    # Train
    model, X_test, y_test, preds, probas, metrics_df, avg_acc = walk_forward_train(feat_df)

    # Backtest
    if model is not None and X_test is not None:
        bt_results, bt_metrics = run_backtest(feat_df, raw, preds, probas, X_test)
    else:
        bt_results, bt_metrics = None, {
            'sharpe': 0, 'max_dd': 0, 'win_rate': 0,
            'total_return': 0, 'market_return': 0, 'trades': 0,
            'buy_signals': 0, 'sell_signals': 0,
            'equity_curve': [], 'market_curve': [],
        }

    signal = get_signal(model, feat_df, raw)
    folds  = metrics_df.to_dict('records') if len(metrics_df) > 0 else []

    result = {
        'ticker'       : short,
        'date'         : str(date.today()),
        'rows'         : len(raw),
        'date_from'    : str(raw.index.min().date()),
        'date_to'      : str(raw.index.max().date()),
        'avg_accuracy' : round(avg_acc, 1),
        'folds'        : folds,
        'backtest'     : bt_metrics,
        'signal'       : signal,
        'technical'    : tech,
        'feature_cols' : FEATURE_COLS,
        'n_features'   : len(FEATURE_COLS),
    }

    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result


if __name__ == '__main__':
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else 'COFORGE'
    r = analyse(t)
    print(json.dumps(r, indent=2))