import os
from dotenv import load_dotenv
import yfinance as yf, pandas as pd, numpy as np, matplotlib.pyplot as plt

# Load key-value pairs from .env
load_dotenv()

# Read .env variables
period = os.getenv("PERIOD")
interval = os.getenv("INTERVAL")

# Portfolio construction 
num_shares = pd.Series({
    "MSFT": 100, 
    "AAPL": 200, 
    "KO": 300, 
})

portfolio = yf.Tickers(num_shares.index.tolist())

# DataFrame creation
df = portfolio.download(period=period, interval=interval)

# Daily log returns
df_closing_prices = df["Close"].dropna()
prev_closing_prices = df_closing_prices.shift(1)

prev_values = prev_closing_prices * num_shares
prev_portfolio_value = prev_values.sum(axis=1)

simple_daily_return = (df_closing_prices/prev_closing_prices) - 1
daily_weight = prev_values.div(prev_portfolio_value, axis=0)

weighted_sum = (simple_daily_return * daily_weight).sum(axis=1)
log_daily_portfolio_return = np.log(1 + weighted_sum)

# Historical CVaR at 95% confidence
return_at_5th_percentile = np.percentile(log_daily_portfolio_return, 5)
mean_5_percentiles = np.mean(log_daily_portfolio_return[log_daily_portfolio_return <= return_at_5th_percentile])

current_portfolio_value = df_closing_prices.iloc[-1] @ num_shares

mean_5_percentiles_in_dollars = current_portfolio_value * (np.exp(mean_5_percentiles) - 1)

print(mean_5_percentiles_in_dollars)