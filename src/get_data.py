import numpy as np
import pandas as pd
import warnings
from pandas.tseries.offsets import CustomBusinessDay
import pandas_market_calendars as mcal
import yfinance as yf
from datetime import datetime


"""
Market data utilities for option pricing and expiry selection.

Functions in this module provide:
- Trading-day calendars (NYSE)
- Expiry handling
- Standardized log returns
- Dataframe with calls and puts data merged
"""

def get_bday_calendar():
    """
    Creates the calendar of the Business days according to NYSE calendar
    Parameters:
    ----------
    none
    
    Returns:
    -------
    The calendar of the Business days according to NYSE calendar
    """
    nyse = mcal.get_calendar("NYSE") #new york stock exchange calendar
    holidays = nyse.holidays().holidays #select the holidays
    return CustomBusinessDay(holidays=holidays) #select the business days

def get_trading_days(expiries, calendar, start_date=None):
    """
    Parameters:
    ----------
    expiries : list of str
        Option expiry dates (YYYY-MM-DD)
    start_date : pd.Timestamp, optional (default: today at midnight)

    Returns:
    -------
    list[int]
        Number of trading days to each expiry
    """
    expiry_dates = [pd.Timestamp(date) for date in expiries] 
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()
    trading_days = [len(pd.date_range(start_date, expiry, freq=calendar)) - 1 for expiry in expiry_dates]
    return trading_days

def get_expiries_options(ticker_opt):
    """
    It returns the available option expiries
    Parameters:
    ----------
    ticker_opt:  yfinance.Ticker 

    Returns:
    -------
    list of str of expiries of the option of the form "YYYY-MM-DD"  
    """
    return ticker_opt.options

def get_standardized_log_returns(ticker_opt, time_span="3000d"):
    """
    It downloads the normalized log returns for the underlying of a given option
    Parameters:
    ----------
    ticker_opt: yfinance.Ticker, the option of which we want the underlying data
    time_span: time window from which to extract the underlying data. Default value: 3000 days
    
    Returns:
    -------
    log returns normalized to the standard normal distribution
    """
    data = ticker_opt.history(period=time_span)["Close"]
    log_returns = np.log((data / data.shift(1)).dropna())
    return (log_returns - log_returns.mean()) / log_returns.std()

def get_hist_volatility(ticker_opt, days_to_expiry, time_span="3000d"):
    """
    It downloads the normalized log returns for the underlying of a given option
    Parameters:
    ----------
    ticker_opt: yfinance.Ticker, the option of which we want the underlying data
    days_to_expiry: number of trading days to use to compute the historical volatility
    time_span: time window from which to extract the underlying data. Default value: 3000 days
    
    Returns:
    -------
    annualized historical volatility
    """
    data = ticker_opt.history(period=time_span)["Close"]
    log_ret = np.log((data / data.shift(1)).dropna())
    sigma_daily = np.std(log_ret[-days_to_expiry:])
    return sigma_daily * np.sqrt(365) #annualized coherently with my conventions 
"""
def get_risk_free_rate_from_yahoo(T):
    It estimates the risk free rate from yahoo data
    Parameters:
    ----------
    T: annualized time scale used to look at risk free rate
    
    Returns:
    -------
    risk free rate 
    if T <= 0.5:
        sym = "^IRX"
    elif T <= 3:
        sym = "^FVX"
    else:
        sym = "^TNX"

    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="1d")
        # take last close; if empty, raise
        if hist.empty:
            raise RuntimeError(f"No data for {sym}")
        yield_percent = hist["Close"].iloc[-1]  
        r = float(yield_percent) / 100.0  # yfinance rates are in percent
        return r
    # fallback: use small default
    except Exception as e:
        warnings.warn(f"Using fallback risk-free rate: {e}")
    return 0.03
"""
    
#it gives a df with the data for calls and puts, in particular it has an estimate of dividends


def get_underlying_price(ticker_opt):
    """
    Parameters:
    ----------
    ticker_opt : The ticker of the option 

    Returns:
    -------
    float
        The today price of the underlying asset
    """
    hist = ticker_opt.history(period="1d")
    if hist.empty:
        raise RuntimeError("No price data available")
    return hist["Close"].iloc[-1]

def annualized_expiry(expiry_date):   
    """
    Parameters:
    ----------
    expiry_date : A date as a string of the form %Y-%m-%d 

    Returns:
    -------
    float
        The annualized time corresponding to a given date
    """
    return (datetime.strptime(expiry_date, "%Y-%m-%d") - pd.Timestamp.today()).days / 365 

def get_option_data(symbol, expiration, S):
    """
    It returns a df with option data for put and call, including mid prices and the ratio price over strike
    Parameters:
    ----------
    symbol:  yfinance.Ticker
    expiration: expiration date of the option
    S: price of the underlying
    
    Returns:
    -------
    df with with option data for put and call
    """
    opt = symbol
    chain = opt.option_chain(expiration)
    table_c = chain.calls 
    table_p = chain.puts
    
    df = table_c.merge(
        table_p,
        on="strike",
        how="inner",
        suffixes=("_call", "_put"))
    
    df["mid_price_put"] = (df["bid_put"] + df["ask_put"])/2
    df["mid_price_call"] = (df["bid_call"] + df["ask_call"])/2
    df["in_the_moneyness"] = S/df["strike"]
    return df
