import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv(".env.backtest")
from data import fetch_log_returns
from analytics import calculate_log_cvar    

def backtest_records() -> pd.DataFrame:
    """
    Backtests the VaR model over historical log returns and records breaches.
    """
    num_shares = pd.Series({
        "MSFT": 100, 
        "AAPL": 200, 
        "KO": 300
    })

    window = int(os.getenv("WINDOW"))
    log_returns, _ = fetch_log_returns(num_shares)

    records = []
    for index in range(len(log_returns) - window):
        log_returns_window = log_returns.iloc[index:index + window]
        window_var, _ = calculate_log_cvar(log_returns_window)
        next_day_date = log_returns.index[index + window]
        next_day_return = log_returns.iloc[index + window]
        breach = next_day_return < window_var

        records.append({
            "date": next_day_date,
            "var": window_var,
            "return": next_day_return,
            "breach": breach
        })

    return pd.DataFrame(records)

def calculate_breach_rate(backtest_records: pd.DataFrame) -> float:
    """
    Calculates the breach rate from backtest records.
    """
    total_breaches = backtest_records["breach"].sum()
    total_records = len(backtest_records)
    return total_breaches / total_records 

def group_records_by_year(backtest_records: pd.DataFrame) -> pd.DataFrame: 
    backtest_records["year"] = backtest_records["date"].dt.year
    yearly = backtest_records.groupby("year")["breach"].agg(["sum", "count"])
    yearly["rate"] = yearly["sum"]/yearly["count"]
    return yearly

def kupiec_pof(period: int, observed_breaches: int, confidence_level: float = 0.95) -> tuple[float, float, bool]: 
    observed_breach_rate = observed_breaches / period
    theoretical_breach_rate = 1 - confidence_level
    lr_pof = -2*np.log(
        np.divide(
            ((1-theoretical_breach_rate)**(period-observed_breaches))*theoretical_breach_rate**observed_breaches, 
            ((1-observed_breach_rate)**(period-observed_breaches))*observed_breach_rate**observed_breaches
        )
    )
    chi_squared = 3.841459  # Chi-squared critical value for 1 degree of freedom at 95% confidence
    
    if lr_pof > chi_squared:
        return lr_pof, chi_squared, False
    else:
        return lr_pof, chi_squared, True
    
yearly_records = group_records_by_year(backtest_records())
period, observed_breaches = yearly_records["count"].sum(), yearly_records["sum"].sum()
kupiec_results = kupiec_pof(period, observed_breaches)
print(kupiec_results)