import pandas as pd
import matplotlib.pyplot as plt

def performance_metrics(data):
    """Calculate and display performance metrics for the strategy."""
    total_return = data['Portfolio_Value'].iloc[-1] - data['Portfolio_Value'].iloc[0]
    annualized_return = (data['Portfolio_Value'].iloc[-1] / data['Portfolio_Value'].iloc[0]) ** (252 / len(data)) - 1
    volatility = data['Strategy_Return'].std() * (252 ** 0.5)
    sharpe_ratio = annualized_return / volatility

    print(f"Total Return: ${total_return:.2f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

def plot_performance(data):
    """Plot the cumulative returns of the strategy."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Portfolio_Value'], label='Strategy Portfolio Value')
    plt.title('Cumulative Returns of Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid()
    plt.savefig('results/performance_analysis.png')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('results/backtest_results.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    performance_metrics(data)
    plot_performance(data)
