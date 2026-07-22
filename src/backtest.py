import pandas as pd
import numpy as np
def run_backtest(predictions, X_test, y_test, df):
    
    results = pd.DataFrame(index=X_test.index)
    results['Actual'] = y_test
    results['Predicted'] = predictions
    results['Open'] = df.loc[X_test.index, 'Open']
    results['Close'] = df.loc[X_test.index, 'Close']
    
    results["daily_returns"] = np.where(results["Predicted"], (results["Close"] - results["Open"]) / results["Open"],(results["Open"] - results["Close"]) / results["Open"])

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