import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

def fetch_stock_data(symbol, start):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start)
    return data["Close"].to_frame()

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # User input
    symbol = input("Enter stock symbol (e.g., 'AAPL'): ").strip()
    start_date = "2020-01-01"
    
    # Fetch data
    data = fetch_stock_data(symbol, start=start_date)
    print(f"\nData from {data.index[0].date()} to {data.index[-1].date()}")
    
    # 1. Simple Linear Regression (True straight laine)
    X_time = np.arange(len(data)).reshape(-1, 1)  # Simple numeric sequence
    y = data['Close'].values
    lr_model = LinearRegression().fit(X_time, y)
    lr_pred = lr_model.predict(X_time)
    
    # 2. LSTM Preparation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    X_lstm, y_lstm = [], []
    for i in range(60, len(scaled_data)):
        X_lstm.append(scaled_data[i-60:i, 0])
        y_lstm.append(scaled_data[i, 0])
    
    X_lstm = np.array(X_lstm).reshape(-1, 60, 1)
    y_lstm = np.array(y_lstm)
    
    # Train LSTM
    lstm_model = build_lstm_model((X_lstm.shape[1], 1))
    lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32, verbose=1)
    
    # LSTM Predictions
    lstm_pred = lstm_model.predict(X_lstm)
    lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
    
    # Align data for plotting and printing
    dates = data.index[60:]  # First 60 days used for initial window
    actual_prices = data['Close'].values[60:]
    lr_prices = lr_pred[60:]
    
    # Create DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': actual_prices,
        'Linear Regression Prediction': lr_prices,
        'LSTM Prediction': lstm_pred
    })
    
    # Print the last 5 predictions
    print("\nLast 5 Predictions:")
    print(predictions_df.tail().to_string(index=False))
    
    # Print next day prediction
    print(f"\nNext Day Prediction ({dates[-1].date()} -> {dates[-1].date() + pd.Timedelta(days=1)}):")
    print(f"Linear Regression: {lr_pred[-1]:.2f}")
    
    # For LSTM next day prediction, we need the last 60 days
    last_60_days = scaled_data[-60:].reshape(1, 60, 1)
    next_day_lstm = scaler.inverse_transform(lstm_model.predict(last_60_days))[0][0]
    print(f"LSTM Prediction: {next_day_lstm:.2f}")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual_prices, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(dates, lr_prices, label='Linear Regression', color='green', linewidth=2, linestyle='--')
    plt.plot(dates, lstm_pred, label='LSTM Prediction', color='red', alpha=0.7)
    
    plt.title(f'{symbol} Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    import os
    import pickle
    from tensorflow.keras.models import save_model

    # ✅ Ensure 'backend' folder exists before saving
    os.makedirs("backend", exist_ok=True)

    # ✅ Save Linear Regression Model
    with open("backend/lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # ✅ Save LSTM Model
    save_model(lstm_model, "backend/lstm_model.h5")

    # ✅ Save Scaler
    with open("backend/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Models saved successfully!")

if __name__ == "__main__":
    main()


