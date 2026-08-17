import numpy as np
import pandas as pd

def calculate_log_cvar(log_returns: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Calculates historical VaR and CVaR for log returns at a given confidence level.
    """
    percentile = (1 - confidence_level) * 100
    var_log = np.percentile(log_returns, percentile)
    cvar_log = np.mean(log_returns[log_returns <= var_log])
    
    return var_log, cvar_log

def log_return_to_dollar_loss(cvar_log: float, portfolio_value: float) -> float:
    """
    Converts log return CVaR into estimated portfolio dollar loss.
    """
    return portfolio_value * (np.exp(cvar_log) - 1)