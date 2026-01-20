from src.get_data import(
    annualized_expiry,
    get_option_data
)

from src.compute_parameters_options import(
    get_risk_free_rate_from_yahoo,
    get_effective_rate
)

"""
Obtain data necessary to price an option. 

Functions in this module provide:
- dataframe with various option data including strikes 
- time to maturity annualized
- Effective rate (risk free rate plus dividend )
"""

def option_data(ticker_opt,expiry ,S):
    """
    It returns options data

    Args:
        ticker_opt: yfinance ticker
        expiry: str date of expiry
        S: asset price

    Returns:
        df: a df with yfinance data for the option with the given expiry 
        T : annualized time to maturity
        r_eff : effective rate
    """
    option = get_option_data(ticker_opt, expiry, S) 
    T = annualized_expiry(expiry)
    r = get_risk_free_rate_from_yahoo(T)
    r_eff = get_effective_rate(S, option, r, T)  #it includes dividend
    return option, T, r_eff

