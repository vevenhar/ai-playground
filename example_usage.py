from crypto_prediction import CryptoPricePredictor
import matplotlib.pyplot as plt

def main():
    predictor = CryptoPricePredictor(crypto_symbol='bitcoin', currency='USD')
    
    print("Fetching historical data...")
    data = predictor.fetch_historical_data(days=60)
    
    if data is not None:
        print(f"Fetched {len(data)} days of historical data")
        
        print("Preparing data...")
        predictor.prepare_data(test_size=0.2, target_days_ahead=1)
        
        print("Training model...")
        predictor.train_model()
        
        print("Evaluating model...")
        metrics = predictor.evaluate_model()
        
        if metrics:
            print("\nModel Evaluation Metrics:")
            print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
            print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
            print(f"R-squared (R²): {metrics['r2']:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        
        print("\nPredicting future prices...")
        predictions = predictor.predict_future_prices(days=7)
        
        if predictions is not None:
            print("\nPredicted Prices for the Next 7 Days:")
            for date, row in predictions.iterrows():
                print(f"{date.date()}: ${row['predicted_price']:.2f}")
        
        print("\nPlotting results...")
        predictor.plot_results(predictions)
        print("Plot saved as 'price_prediction.png'")
        
    else:
        print("Failed to fetch historical data.")

if __name__ == "__main__":
    main()
