import yfinance as yf
import pandas as pd
import os

RAW_PATH = "data/gap_raw.csv"

def load_gap_data(period="10y", force=False):
    """
    Pulls Nifty 50, S&P 500, India VIX, USD/INR.
    Aligns on trading days where ALL four have data.
    Saves to data/gap_raw.csv and returns a merged DataFrame.
    """
    if os.path.exists(RAW_PATH) and not force:
        print(f"Loading cached gap data from {RAW_PATH}")
        df = pd.read_csv(RAW_PATH, index_col="Date", parse_dates=True)
        print(f"  {len(df)} rows, {df.shape[1]} cols")
        return df

    print("Downloading data...")

    tickers = {
        "nifty" : "^NSEI",
        "sp500" : "^GSPC",
        "vix"   : "^INDIAVIX",
        "usdinr": "INR=X",
    }

    frames = {}
    for name, ticker in tickers.items():
        raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        # flatten multi-level columns yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames[name] = raw[['Open', 'Close']].rename(columns={
            'Open' : f'{name}_open',
            'Close': f'{name}_close',
        })
        print(f"  {name}: {len(raw)} rows")

    # merge on index — inner join keeps only days all four have data
    df = frames['nifty']
    for name in ['sp500', 'vix', 'usdinr']:
        df = df.join(frames[name], how='inner')

    df.index.name = "Date"
    df.dropna(inplace=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(RAW_PATH)
    print(f"Saved {len(df)} rows to {RAW_PATH}")
    print(df.tail(3))

    return df


if __name__ == "__main__":
    df = load_gap_data(force=True)
    print(df.head())