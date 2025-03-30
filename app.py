from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Cache models
@lru_cache(maxsize=None)
def load_models():
    lstm_model = load_model('backend/lstm_model.h5')
    lstm_model.compile(optimizer='adam', loss='mse')
    
    with open('backend/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return lstm_model, scaler

def fetch_stock_data(symbol, days=66):
    stock = yf.Ticker(symbol)
    df = stock.history(period=f"{days}d")
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"No data available for {symbol}")
    return df[['Close']]

def generate_plot(symbol, actual, lstm_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, label='Actual Price', color='blue')
    plt.plot(actual.index[-1], lstm_pred, 'ro', markersize=8, label='LSTM Prediction')
    
    # Add today's price annotation
    plt.annotate(f'Today: Rs {actual.values[-1]:.2f}', 
                xy=(actual.index[-1], actual.values[-1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->'))
    
    plt.title(f'{symbol} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['symbol'].upper()
        lstm_model, scaler = load_models()
        data = fetch_stock_data(symbol)
        
        # Get today's actual price
        today_price = data['Close'].iloc[-1]
        
        # LSTM Prediction
        scaled_data = scaler.transform(data.values)
        lstm_input = scaled_data[-60:].reshape(1, 60, 1)
        lstm_value = float(scaler.inverse_transform(lstm_model.predict(lstm_input))[0][0])
        
        plot_url = generate_plot(symbol, data['Close'], lstm_value)
        
        return jsonify({
            'symbol': symbol,
            'actual_price': round(today_price, 2),  # Added current price
            'lstm': round(lstm_value, 2),
            'plot_url': plot_url,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
if __name__ == '__main__':
    os.makedirs("backend", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)