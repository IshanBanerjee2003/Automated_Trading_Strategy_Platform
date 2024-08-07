import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class TradingStrategy:
    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression()

    def generate_features(self):
        """Generate features for the trading model."""
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Close'].rolling(window=10).std()
        self.data['Momentum'] = self.data['Close'] - self.data['Close'].shift(10)
        self.data.dropna(inplace=True)

    def train_model(self):
        """Train the trading model."""
        X = self.data[['Price_Change', 'Volatility', 'Momentum']]
        y = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)
        self.model.fit(X, y)

    def predict(self):
        """Predict market direction using the trained model."""
        X = self.data[['Price_Change', 'Volatility', 'Momentum']]
        self.data['Predictions'] = self.model.predict(X)

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_data.csv')
    strategy = TradingStrategy(data)
    strategy.generate_features()
    strategy.train_model()
    strategy.predict()
    data.to_csv('data/predictions.csv', index=False)
    print("Trading strategy predictions generated.")
