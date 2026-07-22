import pandas as pd

def build_features():
    df = pd.read_csv("data/raw_prices.csv", index_col= "Date", parse_dates=True)

    #
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    df['SMA50'] = df['Close'].rolling(window=50).mean()

    #
    delta = df['Close'].diff()

    gain = delta.clip(0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    RS = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + RS))

    std_ = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['SMA20'] + (2 * std_)
    df['BB_lower'] = df['SMA20'] - (2 * std_)

    daily_return = df['Close'].pct_change()
    df['Volatility_20'] = daily_return.rolling(window=20).std()

    #target label

    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)

    df.dropna(inplace=True)
    df.to_csv("data/features.csv")

    # print(df["Target"].value_counts())

   


    
    
    



    return df




