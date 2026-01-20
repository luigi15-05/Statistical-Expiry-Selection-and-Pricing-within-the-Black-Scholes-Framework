import numpy as np
import pandas as pd
import warnings

"""
Pricing European call options with finite difference method (explicit scheme) and Montecarlo, comparing results
with market mid-prices

Functions in this module provide:
- pricing with finite difference method plus relative error wrt to market mid-prices
- pricing with Montecarlo simulaitons plus relative error wrt to market mid-prices
"""


def bs_finite(S, df, T, r, M, L, PRICE_MAX_RANGE = 2.5 ):
    """
    Implement finite difference method on call option with different strikes and volatilities
    Parameters:
    ----------
    S : asset price
    df: dataframe with option data
    T : time to maturity 
    r : interest rate (same as r_eff in our example)
    M : number of price intervals
    L : number of time intervals
    PRICE_MAX_RANGE : extension of price grid in terms of S (default value 2.5)
    
    Returns:
    -------
    series
        series of prices from finite differences
    """      

    dt = T/L
    dS = PRICE_MAX_RANGE * S/M
    i = np.arange(0, M+1)
    df["finite_diff"] = 0.0
    
    for j in range(len(df)):
        K = df.iloc[j]["strike"]
        sigma = df.iloc[j]["impl_vol"] 
        
        if dt > (dS/(S * sigma))**2:
            warnings.warn(f"the convergence bound is not respected for {sigma}, change the parameters please! ")
            return np.nan 
           
        a = 1/2 * (sigma ** 2 * i * i - r * i) * dt
        b = 1 - (sigma ** 2 * i ** 2 + r) * dt
        c = 1/2 * (sigma ** 2 * i ** 2 + r * i) * dt

        payoff =  np.maximum(i * dS - K, np.zeros(M+1))
        V_store = np.zeros((L+1, M+1))
        V_store[0] = payoff
        
        for k in range(1,L+1): 
            
            V_store[k][0] = 0
            V_store[k][M] = M * dS - K * float(np.exp(-r * (T - k * dt)))
            V_store[k][1:M-1] = a[1:M-1] * V_store[k-1][0:M-2] + b[1:M-1] * V_store[k-1][1:M-1] + c[1:M-1] * V_store[k-1][2:M] 
        
        df.loc[j, "finite_diff"] = V_store[L][int(S/dS)]
    return df["finite_diff"]

def fd_with_error(S, df, T, r, M, L, PRICE_MAX_RANGE = 2.5):
    """
    Implement finite difference method on call df and compute relative error with market mid-prices
    Parameters:
    ----------
    S : asset price
    df: dataframe with option data
    T : time to maturity 
    r : interest rate (same as r_eff in our example)
    M : number of price intervals
    L : number of time intervals
    PRICE_MAX_RANGE : extension of price grid in terms of S (default value 2.5)
    
    Returns:
    -------
    df
        with column of fd_prices and one with relative error
    """     
    df["finite_difference_price_call"] = bs_finite(S, df, T, r,
                                                           M, L, PRICE_MAX_RANGE = 2.5)
    df["finite_diff_rel_error"] = abs((df["finite_difference_price_call"] - df["mid_price_call"]))/df["finite_difference_price_call"]
    return df


def bs_simulation_montecarlo(S, K, T, r, sigma, L, M, D=0, type = "c"):
    """
    Price call option with different strikes and volatilities with (antithetic) Montecarlo. 
    It also give the statistical error. 
    Parameters:
    ----------
    S : asset price
    K : column of strikes
    T : time to maturity 
    r : interest rate (same as r_eff in our example)
    sigma : column of volatilities
    L : number of time intervals
    M : number of simulations

    Returns:
    -------
    series
        series of prices from finite differences
    """         
    L = int(L)
    M = int(M)
    dt = T/L
    #simulation
    t = dt * np.fromfunction(lambda i,j : 1 , (M,L) )
    dX = np.random.normal(0, np.sqrt(dt), size=(M,L))
    
    log_W1 = np.cumsum(sigma[:,None,None] * dX + (r - D - sigma[:,None,None]**2 / 2) * t,axis=2)
    log_W2 = np.cumsum(-sigma[:,None,None] * dX + (r - D - sigma[:,None,None]**2 / 2) * t,axis=2)
    W1 = S * np.exp(log_W1)
    W2 = S * np.exp(log_W2)
    #evaluation
    V1 = np.exp(-r * T) * np.maximum(W1[:,:,-1]- K[:,None], 0)
    V2 = np.exp(-r * T) * np.maximum(W2[:,:,-1]- K[:,None], 0)
    U = (V1+V2)/2
    V = U.mean(axis=1)
    #error
    var = np.var(U, axis=1, ddof=1)
    error = np.sqrt(var / M)
    return V, error

def montercarlo_df(df, S, T, r, L, M):
    """
    Price a df with options data using montecarlo simulations and compute relative errors
    
    Parameters:
    ----------
    df : data framw with options data
    E : column of strikes
    T : time to maturity 
    r : interest rate (same as r_eff in our example)
    sigma : column of volatilities
    L : number of time intervals
    M : number of simulations

    Returns:
    -------
    dataframe with options data plus new columns
        -with Monte Carlo price
        -with Monte Carlo statistical errors
        -with relative errors wrt to market mid-prices
    """         
    
    best_mc_price, best_mc_err = bs_simulation_montecarlo(S,df["strike"].values, T, r,
                                                          df["impl_vol"].values, L, M)
    df["montecarlo_price_call"] = best_mc_price
    df["montecarlo_std_error"] = best_mc_err
    df["montecarlo_rel_error"] = abs((df["montecarlo_price_call"] - df["mid_price_call"]))/df["montecarlo_price_call"]
    return df