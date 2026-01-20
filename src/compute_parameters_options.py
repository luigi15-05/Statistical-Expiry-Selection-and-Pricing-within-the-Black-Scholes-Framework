import numpy as np
from scipy.stats import norm
import warnings
import yfinance as yf

"""
Extract data from options such as implied volatility and effective rates (e.g. risk free rate plus dividends).

Functions in this module provide:
- Root finders based on Newton method plus fallback with bisection (more general than Black-Scholes) 
- Black-Scholes formula for option pricing
- Implied volatilty calculator
- Effective rate calculator from forwards with put-call parity
"""
# ==========================
# Parameters
# ==========================

MAX_ITERATIONS_ROOT_FINDER = 10000       # maximal number of iterations 
TOLERANCE_ROOT_FINDER = 1e-5             # Absolute tolerance for root finding
LOWER_LIMIT_IMPLIED_VOLATILITY = 1e-8    # Lower limit for our implied volatility calculator   
UPPER_LIMIT_IMPLIED_VOLATILITY = 2       # Upper limit for our implied volatility calculator   
DERIVATIVE_PRECISION = 1e-6              # Expected relative magnitude of the increment in derivative computations


def bis_zero_vect(f, x_min, x_max, max_iter = MAX_ITERATIONS_ROOT_FINDER, tol = TOLERANCE_ROOT_FINDER ):
    """
    root finder with bisection method in the interval (x_min, x_max)
    Parameters:
    ----------
    f : The function to solve
    x_min : lower limit of the interval 
    x_max : upper limit of the interval 
    max_iter : maximal number of iterations
    tol : absolute tolerance for the roots

    Returns:
    -------
    array
        An array of the zeros
    """    
    a = np.atleast_1d(x_min).copy()
    b = np.atleast_1d(x_max).copy()
    x0 = (a + b)/2
    check = np.zeros_like(x0, dtype=bool)

    for n in range(max_iter):        
        fa = f(a)
        fb = f(b)
        f0 = f(x0)
        if fa * fb > 0:
            warnings.warn(f"No change sign in this interval, try to reduce it or something else ")
            return np.nan    
        a = np.where(f0 * fa < 0, a, x0 )
        b = np.where(f0 * fb < 0, b, x0)
        x_temp = (a+b)/2
        f_temp = f(x_temp)
        
        check_now = np.abs(f_temp) < tol
        check = check or check_now
        x0[np.invert(check)] = x_temp[np.invert(check)]
        if np.all(np.abs(f_temp) < tol):
            break    
    return x0     
    
def derivative(f, x, h = DERIVATIVE_PRECISION):
    """
    function to compute the derivative
    Parameters:
    ----------
    f : The function to be derived
    x : position 
    h : relative magnitude of the increment

    Returns:
    -------
    float/array-like
        derivative of f in x
    """  
    x = np.atleast_1d(x)
    h_eff = h * np.maximum(1.0, np.abs(x))
    return (f(x + h_eff) - f(x - h_eff)) / (2*h_eff)

def newt_zeros(f, x_min, x_max, max_iter = MAX_ITERATIONS_ROOT_FINDER, tol = TOLERANCE_ROOT_FINDER): 
    """
    root finder with Newton method plus authomatic fallback with bisection method in the interval (x_min, x_max)
    Parameters:
    ----------
    f : The function to solve
    x_min : lower limit of the interval 
    x_max : upper limit of the interval 
    max_iter : maximal number of iterations
    tol : absolute tolerance for the roots

    Returns:
    -------
    array
        An array of the zeros
    """      
    #dx = np.linspace(x_min, x_max, 10000)
    num_points = min(5000, max(100, int((x_max - x_min) * 1000)))
    dx = np.linspace(x_min, x_max, num_points)
    fx = f(dx)    
    sf = np.sign(fx)
    indices = np.where(sf[1:]*sf[:-1]<0)[0]
    if indices.size == 0:
        warnings.warn("No zero found in this interval, try something else ")
        return np.array([]) 
    x0 = (dx[indices]+dx[indices+1])/2
    u = dx[indices]
    v = dx[indices+1]
    converged = np.zeros_like(x0, dtype=bool)
        
    for _ in range(max_iter):
    
        dfx = derivative(f, x0)
        mask = (np.abs(dfx) < 1e-14) | np.abs((f(x0)/dfx) > (x_max - x_min)/1e2)
        ind_cp = np.where(mask)[0] 
        x_crit = x0[ind_cp]
        
        if x_crit.size > 0:
            
            xm = u[ind_cp]
            xp = v[ind_cp]
            f_crit = f(x_crit)
            fp = f(xp)
            mask_sign = (f_crit * fp < 0)

            left  = np.where(mask_sign, x_crit, xm)
            right = np.where(mask_sign, xp, x_crit)
            zero_bis = bis_zero_vect(f, left, right)
            warnings.warn(f"{zero_bis} required bisection as some derivatives was too small ")
        else:
            zero_bis = np.array([])
        #create array to filter points for Newton (keep) and points to apply bisection
        keep = np.ones_like(x0, dtype=bool)
        keep[ind_cp] = False
        x0 = x0[keep]
        dfx = dfx[keep]
        converged = converged[keep]
        u = u[keep]
        v = v[keep]
        
        x_new = x0 - f(x0) / dfx
                
        #convergence check
                
        converged_now = np.abs(f(x_new)) < tol
        converged = converged | converged_now
        x0[np.invert(converged)] = x_new[np.invert(converged)]
        if np.all(np.abs(f(x0)) < tol):
            break  
    
    if zero_bis.size > 0:    
        return np.concatenate([x0, zero_bis])
    else:
        return x0

def bs_formula(S, K, T, r, sigma, D=0, option_type = "c"):
    """
    Implement Black-Scholes formula for call or put
    Parameters:
    ----------
    S : asset price
    K : strike 
    T : time to maturity 
    r : risk-free rate
    sigma : volatility
    D : dividend, 0 default value, 
    type = "c" (default) for call, "p" for put

    Returns:
    -------
    float
        option price
    """      
    if S <= 0 or K <= 0 or T <= 0:
        warnings.warn("Invalid input parameters")
        return np.nan
    d1 = (np.log(S/K) + (r - D + 1/2 * sigma * sigma) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "c":
        return S * np.exp(- D * T) * norm.cdf(d1,0,1) - K * np.exp(- r * T) * norm.cdf(d2,0,1) 
    if option_type == "p":
        return K * np.exp(- r * T) * norm.cdf(-d2,0,1) - S * np.exp(- D * T) * norm.cdf(-d1,0,1)

def implied_vol_single(mid_price, S, K, T, r_eff,
                       a = LOWER_LIMIT_IMPLIED_VOLATILITY, b = UPPER_LIMIT_IMPLIED_VOLATILITY):
    """
    Use my Newton root finder to compute implied volatility on a single option 
    Parameters:
    ----------
    mid_price  : dataframe with option data
    S : asset price
    K : strike 
    T : time to maturity 
    r_eff : effective rate
    a : lower bound on implied volatility 
    b = upper bound on implied volatility 
    Returns:
    -------
    float
        option price
    """          
    intrinsic = max(S - K * np.exp(-r_eff * T), 0) #theoretical value of an option if volatility was 0. 
    #Intrinsic is the minimal value of an option, S the maximal. If price out of these values, the implied vol doesn't exist
    if mid_price < intrinsic or mid_price > S:
        warnings.warn("The price is unrealistic. Check if everything is correct ")
        return np.nan
    res = newt_zeros(lambda x: bs_formula(S, K, T, r_eff, x) - mid_price, a, b)
    return res[0] if len(res) else np.nan


def compute_implied_vol(option_df, S, T, r_eff,
                        a = LOWER_LIMIT_IMPLIED_VOLATILITY, b = UPPER_LIMIT_IMPLIED_VOLATILITY):
    """
    Apply my implied volatility calculator to the option dataframe 
    Parameters:
    ----------
    option_df  : dataframe with option data
    S : asset price
    T : time to maturity 
    r_eff : effective rate
    a : lower bound on implied volatility 
    b = upper bound on implied volatility 
    Returns:
    -------
    series column
        with the implied volatility
    """   
    iv = option_df.apply(lambda row: implied_vol_single(row["mid_price_call"],
                                                           S, row["strike"], T, r_eff, a, b), axis=1)
    df_iv = option_df.assign(impl_vol=iv)
    return df_iv
   
    #return option_df.apply(lambda row: implied_vol_single(row["mid_price_call"],
    #                                                       S, row["strike"], T, r_eff, a, b), axis=1)



def get_risk_free_rate_from_yahoo(T):
    """
    It estimates the risk free rate from yahoo data
    Parameters:
    ----------
    T: annualized time scale used to look at risk free rate
    
    Returns:
    -------
    risk free rate 
    """
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


def get_effective_rate(S, option_df, r, T):
    """
    Extract effective rate from option data using put call parity
    Parameters:
    ----------
    S : asset price
    option_df  : dataframe with option data
    r : risk free rate 
    T : time to maturity 
    
    Returns:
    -------
    float
        effective rate (risk free plus dividends)
    """ 
    df_atm = option_df[np.abs(option_df["strike"]/S - 1) < 0.02]
    if len(df_atm ) == 0:
        warnings.warn("it wasn't possible to extract an effective rate because no option is atm ")
        return np.nan
    F = np.mean(df_atm["strike"] + np.exp(r*T)*(df_atm["mid_price_call"] - df_atm["mid_price_put"]))    
    return (1/T) * np.log(F/S)