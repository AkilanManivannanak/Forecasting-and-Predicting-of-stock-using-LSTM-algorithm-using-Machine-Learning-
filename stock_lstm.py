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

time_step = 60
# Rows used as training targets: the first 80% of the windows (see step 4).
split_row = int((len(data) - time_step) * 0.8) + time_step

# 2. Data Preprocessing: Normalize the 'Close' prices to range (0,1).
# Fit the scaler on the training period only; fitting on all rows would leak the test
# period's min/max into training.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[:split_row])
scaled_data = scaler.transform(data)

# 3. Feature Engineering: Create time-series sequences for training
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step)
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

# 4. Split into Training and Testing datasets (80/20, chronological)
split = split_row - time_step  # same boundary the scaler was fit on
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

# Baseline: predict tomorrow's close = today's close. An LSTM on raw prices is only useful
# if it beats this.
naive = scaler.inverse_transform(X_test[:, -1, 0].reshape(-1, 1))
naive_rmse = math.sqrt(mean_squared_error(y_test_actual, naive))
print(f'Naive baseline RMSE (tomorrow = today): {naive_rmse:.4f}')
print(f'LSTM vs naive: {"better" if rmse < naive_rmse else "worse"} by {abs(naive_rmse - rmse):.4f}')

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
