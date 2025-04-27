# ai-playground
AI playground for experimenting with machine learning models and algorithms.

## Cryptocurrency Price Prediction

This project implements a linear regression mechanism to predict cryptocurrency prices based on historical data.

### Features

- Fetches historical cryptocurrency price data from CoinGecko API
- Implements linear regression model for price prediction
- Creates features from time series data (lag features, rolling statistics)
- Evaluates model performance using various metrics
- Predicts future prices for a specified number of days
- Visualizes historical and predicted prices

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

```python
from crypto_prediction import CryptoPricePredictor

# Initialize the predictor for Bitcoin
predictor = CryptoPricePredictor(crypto_symbol='bitcoin', currency='USD')

# Fetch historical data
data = predictor.fetch_historical_data(days=60)

# Prepare data, train and evaluate the model
predictor.prepare_data()
predictor.train_model()
metrics = predictor.evaluate_model()

# Predict future prices
predictions = predictor.predict_future_prices(days=7)

# Plot results
predictor.plot_results(predictions)
```

For a complete example, see `example_usage.py`.

### Example

Run the example script:

```bash
python example_usage.py
```

This will:
1. Fetch 60 days of historical Bitcoin price data
2. Train a linear regression model
3. Evaluate the model performance
4. Predict Bitcoin prices for the next 7 days
5. Generate a visualization saved as 'price_prediction.png'

### Limitations

- Linear regression is a simple model and may not capture complex market dynamics
- Cryptocurrency prices are highly volatile and influenced by many external factors
- Past performance is not indicative of future results
