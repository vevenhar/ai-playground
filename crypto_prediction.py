import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta

class CryptoPricePredictor:
    """
    A class for predicting cryptocurrency prices using linear regression.
    """
    
    def __init__(self, crypto_symbol='BTC', currency='USD'):
        """
        Initialize the predictor with the cryptocurrency symbol and currency.
        
        Parameters:
        -----------
        crypto_symbol : str
            The symbol of the cryptocurrency (default: 'BTC' for Bitcoin)
        currency : str
            The currency to convert to (default: 'USD')
        """
        self.crypto_symbol = crypto_symbol
        self.currency = currency
        self.model = LinearRegression()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def fetch_historical_data(self, days=30):
        """
        Fetch historical price data for the specified cryptocurrency.
        
        Parameters:
        -----------
        days : int
            Number of days of historical data to fetch (default: 30)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the historical price data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        url = f"https://api.coingecko.com/api/v3/coins/{self.crypto_symbol.lower()}/market_chart/range"
        params = {
            'vs_currency': self.currency.lower(),
            'from': start_timestamp // 1000,  # Convert to seconds
            'to': end_timestamp // 1000       # Convert to seconds
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            
            prices = data.get('prices', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            
            for i in range(1, 6):
                df[f'price_lag_{i}'] = df['price'].shift(i)
            
            df['rolling_mean_3'] = df['price'].rolling(window=3).mean()
            df['rolling_std_3'] = df['price'].rolling(window=3).std()
            
            df.dropna(inplace=True)
            
            self.data = df
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def prepare_data(self, test_size=0.2, target_days_ahead=1):
        """
        Prepare the data for training and testing.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the data to use for testing (default: 0.2)
        target_days_ahead : int
            Number of days ahead to predict (default: 1)
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            print("No data available. Please fetch historical data first.")
            return None
        
        self.data['target'] = self.data['price'].shift(-target_days_ahead)
        self.data.dropna(inplace=True)
        
        features = ['price', 'day_of_week', 'day_of_month', 'month', 
                   'price_lag_1', 'price_lag_2', 'price_lag_3', 
                   'rolling_mean_3', 'rolling_std_3']
        
        X = self.data[features]
        y = self.data['target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """
        Train the linear regression model.
        
        Returns:
        --------
        sklearn.linear_model.LinearRegression
            Trained linear regression model
        """
        if self.X_train is None or self.y_train is None:
            print("Training data not available. Please prepare data first.")
            return None
        
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test data.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        if self.X_test is None or self.y_test is None:
            print("Test data not available. Please prepare data first.")
            return None
        
        y_pred = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def predict_future_prices(self, days=7):
        """
        Predict future prices for the specified number of days.
        
        Parameters:
        -----------
        days : int
            Number of days to predict into the future (default: 7)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the predicted prices
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        if self.data is None:
            print("No data available. Please fetch historical data first.")
            return None
        
        last_data = self.data.iloc[-1:].copy()
        
        future_dates = [last_data.index[0] + timedelta(days=i+1) for i in range(days)]
        predictions_df = pd.DataFrame(index=future_dates, columns=['predicted_price'])
        
        current_data = last_data.copy()
        
        for i in range(days):
            features = ['price', 'day_of_week', 'day_of_month', 'month', 
                       'price_lag_1', 'price_lag_2', 'price_lag_3', 
                       'rolling_mean_3', 'rolling_std_3']
            
            X_pred = current_data[features]
            
            predicted_price = self.model.predict(X_pred)[0]
            
            predictions_df.iloc[i]['predicted_price'] = predicted_price
            
            next_date = future_dates[i]
            new_row = current_data.copy()
            new_row.index = [next_date]
            
            new_row['price'] = predicted_price
            new_row['day_of_week'] = next_date.dayofweek
            new_row['day_of_month'] = next_date.day
            new_row['month'] = next_date.month
            
            new_row['price_lag_1'] = current_data['price'].values[0]
            new_row['price_lag_2'] = current_data['price_lag_1'].values[0]
            new_row['price_lag_3'] = current_data['price_lag_2'].values[0]
            
            prices = [current_data['price_lag_2'].values[0], 
                     current_data['price_lag_1'].values[0], 
                     current_data['price'].values[0]]
            new_row['rolling_mean_3'] = np.mean(prices)
            new_row['rolling_std_3'] = np.std(prices)
            
            current_data = new_row.copy()
        
        return predictions_df
    
    def plot_results(self, predictions_df=None):
        """
        Plot the historical prices and predictions.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing the predicted prices (default: None)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        if self.data is None:
            print("No data available. Please fetch historical data first.")
            return None
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.data.index, self.data['price'], label='Historical Prices', color='blue')
        
        if predictions_df is not None:
            plt.plot(predictions_df.index, predictions_df['predicted_price'], 
                    label='Predicted Prices', color='red', linestyle='--')
        
        plt.title(f'{self.crypto_symbol}/{self.currency} Price Prediction')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({self.currency})')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('price_prediction.png')
        
        return plt.gcf()
