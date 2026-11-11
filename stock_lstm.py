import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# 1. Data Collection: Download historical stock prices
stock_symbol = 'AAPL'  # Example stock symbol
df = yf.download(stock_symbol, start='2015-01-01', end='2024-01-01')
data = df[['Close']]  # Using 'Close' price for prediction

# 2. Data Preprocessing: Normalize the 'Close' prices to range (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Feature Engineering: Create time-series sequences for training
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

# 4. Split into Training and Testing datasets (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Model Development: Build advanced LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(100),
    Dense(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Model Training
model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1)

# 7. Model Validation and Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate with RMSE
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# 8. Visualization of Real vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Stock Price')
plt.plot(predictions, label='Predicted Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
