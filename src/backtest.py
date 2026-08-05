import pandas as pd
import numpy as np
def run_backtest(predictions, X_test, y_test, df):
    
    results = pd.DataFrame(index=X_test.index)
    results['Actual'] = y_test
    results['Predicted'] = predictions
    # Use NEXT day's Open/Close — signal fires after today's close,
    # trade executes tomorrow Open → tomorrow Close
    next_open = df['Open'].shift(-1)
    next_close = df['Close'].shift(-1)

    results['Open'] = next_open.loc[X_test.index]
    results['Close'] = next_close.loc[X_test.index]

    # Drop last row — shift(-1) makes it NaN (no "tomorrow" exists for the final row)
    results.dropna(subset=['Open', 'Close'], inplace=True)

    results["daily_returns"] = np.where(
        results["Predicted"],
        (results["Close"] - results["Open"]) / results["Open"],   # BUY: long
        (results["Open"] - results["Close"]) / results["Open"]    # SELL: short
    )

    results["Cumulative_Strategy"] = (1 + results['daily_returns']).cumprod()
    results["Cumulative_Market"] = (1 + (results['Close'] - results['Open']) / results['Open']).cumprod()

    # print(results[['daily_returns', 'Cumulative_Strategy', 'Cumulative_Market']].tail(10))

    # Sharpe Ratio:
    sharpe = results['daily_returns'].mean() / results['daily_returns'].std() * np.sqrt(252)

    # Max Drawdown:
    rolling_max = results['Cumulative_Strategy'].cummax()
    drawdown = results['Cumulative_Strategy'] / rolling_max - 1
    max_drawdown = drawdown.min()

    # Win Rate:

    win_rate = (results['daily_returns'] > 0).mean()
    

    # print(sharpe, max_drawdown, win_rate)

    return results, sharpe, max_drawdown, win_rate