import pandas as pd
import numpy as np

def build_features():
    df = pd.read_csv("data/raw_prices.csv", index_col="Date", parse_dates=True)

    # ── Existing features ─────────────────────────────────────────────────────

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    delta    = df['Close'].diff()
    gain     = delta.clip(0)
    loss     = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    RS       = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + RS))

    std_         = df['Close'].rolling(window=20).std()
    df['BB_upper']    = df['SMA20'] + (2 * std_)
    df['BB_lower']    = df['SMA20'] - (2 * std_)
    df['BB_width']    = (df['BB_upper'] - df['BB_lower']) / df['SMA20']  # normalised width

    daily_return      = df['Close'].pct_change()
    df['Volatility_20'] = daily_return.rolling(window=20).std()

    # ── New features ──────────────────────────────────────────────────────────

    # MACD — trend momentum
    ema12          = df['Close'].ewm(span=12, adjust=False).mean()
    ema26          = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']     = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']   # positive = bullish momentum

    # ATR — regime volatility (tells model how choppy the market is)
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low']  - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # Day of week — Mon=0, Thu=3 (F&O expiry), Fri=4
    df['day_of_week'] = df.index.dayofweek.astype(float)

    # ── Target ───────────────────────────────────────────────────────────────
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)

    df.dropna(inplace=True)
    df.to_csv("data/features.csv")

    return df