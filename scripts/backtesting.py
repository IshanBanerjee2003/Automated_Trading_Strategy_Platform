import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(data):
    """Backtest the trading strategy."""
    initial_capital = 10000.0
    data['Position'] = data['Predictions'].shift(1)
    data['Strategy_Return'] = data['Position'] * data['Close'].pct_change()
    data['Portfolio_Value'] = initial_capital * (1 + data['Strategy_Return'].cumsum())
    return data['Portfolio_Value']

def plot_results(data):
    """Plot the backtesting results."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Portfolio_Value'], label='Strategy Portfolio Value')
    plt.title('Trading Strategy Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid()
    plt.savefig('results/performance_analysis.png')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('data/predictions.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    portfolio_values = backtest_strategy(data)
    data['Portfolio_Value'] = portfolio_values
    data.to_csv('results/backtest_results.csv')
    plot_results(data)
    print("Backtesting completed and results saved.")
