# Stock price forecasting with an LSTM (AAPL)

A two-layer LSTM that predicts the next day's closing price of a stock from the previous 60 closes, trained on daily data from Yahoo Finance.

> Educational project. Not financial advice, and not a trading strategy.

## How it works

| Step | What happens |
|---|---|
| Data | `yfinance` downloads daily closes for `AAPL`, 2015-01-01 to 2024-01-01 |
| Split | Chronological: first 80% of the 60-day windows for training, last 20% for testing |
| Scaling | `MinMaxScaler` fit on the training period only, then applied to everything (no test leakage) |
| Model | LSTM(100) → LSTM(100) → Dense(50) → Dense(1), Adam, MSE loss, 20 epochs |
| Evaluation | Test RMSE in dollars, compared with a naive baseline (tomorrow's close = today's close) |

## Run it

```bash
pip install -r requirements.txt
python stock_lstm.py
```

It prints the LSTM test RMSE, the naive-baseline RMSE and which one is better, then plots actual vs predicted prices. Change `stock_symbol` and the date range at the top of `stock_lstm.py` to try other tickers.

## Reading the results

Daily prices are close to a random walk, so "tomorrow = today" is a strong baseline. An LSTM trained on raw price levels often lands close to it, so check the baseline line before drawing conclusions from the RMSE.

## Known limitations

- Predicts price levels; test prices above the training range fall outside the scaler's [0, 1] range, which LSTMs extrapolate poorly. Modelling returns is the usual fix.
- One chronological split; no walk-forward validation.
- Single feature (close price); no volume or market context.
