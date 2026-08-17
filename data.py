import os
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def fetch_log_returns(num_shares: pd.Series):
    """
    Downloads historical close prices and computes portfolio daily log returns.
    """
    period = os.getenv("PERIOD", "1y")
    interval = os.getenv("INTERVAL", "1d")
    
    tickers = num_shares.index.tolist()
    df = yf.download(tickers, period=period, interval=interval)
    
    closing_prices = df["Close"].dropna()
    prev_closing_prices = closing_prices.shift(1)
    
    # Portfolio dynamic weights based on previous day prices
    prev_values = prev_closing_prices * num_shares
    prev_portfolio_value = prev_values.sum(axis=1)
    
    simple_daily_return = (closing_prices / prev_closing_prices) - 1
    daily_weight = prev_values.div(prev_portfolio_value, axis=0)
    
    # Weighted sum converted to daily portfolio log return
    weighted_sum = (simple_daily_return * daily_weight).sum(axis=1)
    log_daily_portfolio_return = np.log(1 + weighted_sum)
    
    # Current portfolio total market value for dollar scaling
    current_portfolio_value = closing_prices.iloc[-1] @ num_shares
    
    return log_daily_portfolio_return.dropna(), current_portfolio_value