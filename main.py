from src.data_loader import load_data
from src.features import build_features
from src.visualizer import plot_features, plot_backtest
from src.model import train_model
from src.backtest import run_backtest
from src.reporter import generate_report


load_data()

df = build_features()


plot_features(df)


model, predictions, X_test, y_test, accuracy = train_model(df)

results, sharpe, max_drawdown, win_rate = run_backtest(predictions, X_test, y_test, df)
plot_backtest(results)


output = generate_report(sharpe, max_drawdown, win_rate, accuracy, results["Cumulative_Strategy"].iloc[-1], results["Cumulative_Market"].iloc[-1])
print(output)