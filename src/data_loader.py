import yfinance as yf



def load_data():
    data = yf.download("^NSEI", period="10y", interval="1d")

    data.columns = [f"{metric}" for metric, ticker  in data.columns]
    
    data = data.dropna()
    
    data.to_csv("data/raw_prices.csv")

    return data

