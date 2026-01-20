import numpy as np
import scipy.stats as stats
from scipy.stats import kurtosis

"""
Statistical filters and logic for a data-driven choice of the best expiry for test pricing within the Black-Scholes framework

Functions in this module provide:
- Shapiro-Wilk test to check the normal behavior of log-returns
- Kurtosis test to probe the tail behavior 
- A test to verify the variance scaling behavior of the Brownian motion
- A function combining the test to extract the best expiry
"""

# ==========================
# Parameters
# ==========================

SHAPIRO_ALPHA = 0.05          # Significance level for normality test --> 5% confidence level
MAX_EXCESS_KURTOSIS = 1.0     # Tolerance for heavy tails (excess kurtosis)


def stat_test(lst):
    """
    Parameters:
    ----------
    lst : a list/array/series of numerical data 

    Returns:
    -------
    float
        p-value of the Shapiro test if the list is longer than 10 and shorter than 5000, nan otherwise
    """    
    if not 10 < len(lst) < 5000:
        return np.nan
    _, p_value_sw = stats.shapiro(lst)
    return p_value_sw 
        
    
def get_shap_filter(log_ret_std, opt_trading_days):
    """
    Parameters:
    ----------
    log_ret_std : standardized log returns
    opt_trading_days: list of days of the time windows to analyze

    Returns:
    -------
    list[float]
        list of Shapiro p-value of the log returns for any time windows  
    """        
    return [stat_test(log_ret_std.iloc[-days:]) for days in opt_trading_days]

def variance_scaling_comp(sr, lags=(1,5,10)): 
    """
    Parameters:
    ----------
    sr : a series of data expected to follow a Brownian motion 
    lags: Time lags on which we test the variance. Default values are: 1,5,10
    
    Returns:
    -------
    float
        Variance scaling test of the Brownian motion (0 ideal BM, the bigger the larger deviations from BM)
    """    
    scaled_vars = []
    for l in lags: 
        y = sr[:(len(sr)//l) * l].values.reshape(-1, l).sum(axis=1) 
        scaled_vars.append(np.var(y) / l) 
    return np.std(scaled_vars) 

def get_filter_variance(log_ret_std, opt_trading_days):
    """
    Parameters:
    ----------
    log_ret_std : standardized log returns
    opt_trading_days: list of days of the time windows to analyze
    
    Returns:
    -------
    list[float]
        list of values for the test of Variance scaling of the log returns for any time windows  
    """
    return [variance_scaling_comp(log_ret_std.iloc[-days:]) for days in opt_trading_days]

def kurtosis_test(lst):
    """
    Parameters:
    ----------
    lst: a list/array/series of numerical data
    
    Returns:
    -------
    float
        the kurtosis - 3 of the list
    """
    return ((abs(kurtosis(lst, fisher=True))))

def get_kurtosis_filter(log_ret_std, opt_trading_days):
    """
    Parameters:
    ----------
    log_ret_std : standardized log returns
    opt_trading_days: list of days of the time windows to analyze
    
    Returns:
    -------
    list[float]
        list of values for the test of Variance scaling of the log returns for any time windows  
    """
    return [kurtosis_test(log_ret_std.iloc[-days:]) for days in opt_trading_days]

def get_best_expiry(filter_kurtosis, filter_normal,filter_std, expiries,
                    tol_norm = SHAPIRO_ALPHA, tol_kurt = MAX_EXCESS_KURTOSIS):
    """
    Parameters:
    ----------
    filter_kurtosis : list of kurtosis test results ordered by date
    filter_normal : list of Shapiro test results ordered by date
    filter_std : list of Variance scaling test results ordered by date
    expiries : [str] of all the expiries to analyze
    tol_norm : lower limit for Shapiro test
    tol_kurt : upper limit for kurtosis test
    
    Returns:
    -------
    str
        The expiry date closer to the Black-Scholes hypothesis for the underlying (format %Y-%m-%d )

    Logic:
    We filter expiries by a strict lower limit for the Shapiro test and an an upper bound on the absolute excess kurtosis. 
    Among the filtered ones, we choose the one with the best scaling variance behavior
    """
    mask = [
        (not np.isnan(filter_normal[n])) and
        (filter_normal[n] > tol_norm) and
        (filter_kurtosis[n] < tol_kurt) 
        for n in range(len(expiries))
    ]
    if not any(mask):
        raise ValueError("No expiry satisfies normality and kurtosis constraints")
    reduced_filter_std = [filter_std[n] if mask[n] 
                              else max(filter_std) for n in range(len(expiries))]
    best_value = np.argmin(np.array(reduced_filter_std))

    return expiries[best_value], filter_normal[best_value], filter_std[best_value], filter_kurtosis[best_value]

def get_worst_expiry(filter_kurtosis, filter_normal,filter_std, expiries,
                    tol_norm = SHAPIRO_ALPHA, tol_kurt = MAX_EXCESS_KURTOSIS):
    """
    Parameters:
    ----------
    filter_kurtosis : list of kurtosis test results ordered by date
    filter_normal : list of Shapiro test results ordered by date
    filter_std : list of Variance scaling test results ordered by date
    expiries : [str] of all the expiries to analyze
    tol_norm : lower limit for Shapiro test
    tol_kurt : upper limit for kurtosis test
    
    Returns:
    -------
    str
        The expiry date closer to the Black-Scholes hypothesis for the underlying (format %Y-%m-%d )

    Logic:
    We filter expiries by a strict lower limit for the Shapiro test and an an upper bound on the absolute excess kurtosis. 
    Among the filtered ones, we choose the one with the best scaling variance behavior
    """
    mask = [
        (not np.isnan(filter_normal[n])) and
        (filter_normal[n] < tol_norm) and
        (filter_kurtosis[n] > tol_kurt) 
        for n in range(len(expiries))
    ]
    if not any(mask):
        raise ValueError("All expiries satisfy normality and kurtosis test")
    reduced_filter_std = [filter_std[n] if mask[n] 
                              else min(filter_std) for n in range(len(expiries))]
    worst_value = np.argmax(np.array(reduced_filter_std))

    return expiries[worst_value], filter_normal[worst_value], filter_std[worst_value], filter_kurtosis[worst_value]