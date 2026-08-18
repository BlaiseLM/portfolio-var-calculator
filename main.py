import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env.single")
from data import fetch_log_returns
from analytics import calculate_log_cvar, log_return_to_dollar_loss

def main():
    num_shares = pd.Series({
        "MSFT": 100, 
        "AAPL": 200, 
        "KO": 300, 
    })

    # 1. Fetch log returns
    log_portfolio_returns, current_value = fetch_log_returns(num_shares)

    # 2. Compute log CVaR
    var_95_log, cvar_95_log = calculate_log_cvar(log_portfolio_returns, confidence_level=0.95)

    # 3. Scale to current dollar value
    cvar_95_dollars = log_return_to_dollar_loss(cvar_95_log, current_value)

    print("\n=== Log Return CVaR Metrics ===")
    print(f"Current Portfolio Value: ${current_value:,.2f}")
    print(f"95% 1-Day Log VaR:       {var_95_log:.4%}")
    print(f"95% 1-Day Log CVaR:      {cvar_95_log:.4%}")
    print("-" * 35)
    print(f"Estimated Dollar Impact:  ${cvar_95_dollars:,.2f}\n")

if __name__ == "__main__":
    main()