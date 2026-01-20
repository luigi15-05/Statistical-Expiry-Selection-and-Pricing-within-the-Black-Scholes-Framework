import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
"""
Produce several relevant graphs related to the choice of the expiry, option data, and the pricing

Functions in this module provide:
- Plot Shapiro p value vs time
- Plot variance vs time
- Plot kurtosis excess vs time
- Plot p-value, variance, kurtosis excess together vs time 
- For a chosen date, qq plot of returns and corresponding histogram of log-returns vs Gaussian distribution
- Plot implied volatility vs strikes (smile)
- Plot Monte Carlo and finite difference price vs mid price
- Plot Monte Carlo and finite difference relative errors
"""
   
def pvalue_vs_days(days, filter, my_best_day, my_worst_day, title, x_name, y_name):
    """
    Plot Shapiro p value vs time
    Parameters:
    ----------
    days : list of days
    filter : list of p-values
    my_day : the chosen expiry, which will be highlited
    title : title of the plot
    x_name : name of the x-axis
    y_name : name of the y-axis
    """     
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(days, filter, marker="o")
    plt.title(title)
    ax.set_xscale("log")
    ax.axvline(my_best_day, color="black", linestyle=":", lw=2, label="best expiry" )
    ax.axvline(my_worst_day, color="orange", linestyle=":", lw=2, label="worst expiry" )

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.axhline(0.05, color="red", linestyle="--", label="5% significance")
    plt.legend()
    plt.tight_layout() 
    plt.show()        

def variance_vs_time(std_returns, days, best_expiry, worst_expiry):
    """
    Plot variance vs time
    Parameters:
    ----------
    std_returns : list of variances 
    days : list of days
    my_expiry : the chosen expiry, which will be highlited
    """     
    norm_scaling_var = [np.abs(std_returns.rolling(day).sum().dropna().var() - day) / day for day in days if day > 5]    
    my_days = [day for day in days if day > 5]
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xscale("log")
    ax.set_xlabel("Days to expiry Δt (log scale)")
    ax.plot(my_days, norm_scaling_var, "o-")
    
    ax.axvline(best_expiry, color="black", linestyle=":", lw=2, label="best expiry" )
    ax.axvline(worst_expiry, color="orange", linestyle=":", lw=2, label="worst expiry" )

    ax.set_ylabel(r"$|Var(\Delta t) - \Delta t| / \Delta t$")
    ax.set_title("Variance scaling error vs time scale")
    ax.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def kurtosis_vs_days(days, filter, my_best_day, my_worst_day, title, x_name, y_name):
    """
    Plot excess kurtosis vs time
    Parameters:
    ----------
    days : list of days
    filter : list of p-values
    my_day : the chosen expiry, which will be highlited
    title : title of the plot
    x_name : name of the x-axis
    y_name : name of the y-axis
    """     
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(days, filter, marker="o")
    plt.title(title)
    ax.set_xscale("log")
    ax.axvline(my_best_day, color="black", linestyle=":", lw=2, label="best expiry" )
    ax.axvline(my_worst_day, color="orange", linestyle=":", lw=2, label="worst expiry" )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.axhline(1, color="red", linestyle="--", label="Rejecting level = 1")
    plt.legend()
    plt.tight_layout() 
    plt.show()
    
def get_qq_plot(time_series, days, name):
    """
    Plot qq plot and histogram vs Gaussian
    Parameters:
    ----------
    time_series : list of data to be analyzed
    days : the date to analyze
    name : insert at str to complete the title 
    """    
    #pd.to_datetime(expiry)
    
    lst = time_series.iloc[-days:]
    #lst = time_series.loc[expiry:]
    n = len(lst)
    bin_width = 2 * stats.iqr(lst,) / np.cbrt(n)
    n_bins = 3 * int((lst.max() - lst.min()) / bin_width)
    fig, my_stats = plt.subplots(1,2,figsize = (20,5))

    my_stats[0].hist(lst, bins=n_bins, density=True)
    x = np.linspace(lst.min(), lst.max(), 400)
    y = stats.norm.pdf(x)
    my_stats[0].plot(x,y)
    plt.title(f"""Normalized histogram for the {name}""")
    plt.xlabel("Log-return-normalized")
    plt.ylabel("Density")

    stats.probplot(lst.sort_values(), dist="norm", plot=my_stats[1])
    plt.title(f"""Q-Q plot of log-returns for the {name}""")
    plt.show()

def plot_iv_smile(df, expiry):
    """
    Plot implied volatility vs strike (smile)
    Parameters:
    ----------
    df : dataframe with option data
    expiry : the maturity
    """    
    df_plot = df.dropna(subset=["in_the_moneyness", "impl_vol"])    
    plt.figure(figsize=(8,5))
    plt.scatter(df_plot["in_the_moneyness"], df_plot["impl_vol"], s=20)
    plt.xlabel("Moneyness S/K")
    plt.ylabel("Implied volatility")
    plt.title(f"Implied Volatility Smile for the expiry: {expiry}")
    plt.grid(True)
    plt.show()

def plot_pricing_comparison(df, expiry):
    """
    Plot Monte Carlo and finite difference prices vs mid-price
    Parameters:
    ----------
    df : dataframe with option data
    expiry : the maturity
    """  
    df_plot = df.dropna()
    plt.figure(figsize=(9,6))
    plt.plot(df_plot["mid_price_call"], df_plot["finite_difference_price_call"], label="Finite Difference")
    plt.plot(df_plot["mid_price_call"], df_plot["montecarlo_price_call"], label="Monte Carlo")
    plt.xlabel("real price")
    plt.ylabel("model price")
    plt.title(f"Pricing vs Real Prices for the expiry: {expiry}")
    plt.legend()
    plt.grid(True)
    plt.show()
   
def plot_pricing_errors(df, expiry):
    """
    Plot Monte Carlo and finite difference relative errors (wrt mid-price)
    Parameters:
    ----------
    df : dataframe with option data
    expiry : the maturity
    """  
    df_plot = df.dropna()
    plt.figure(figsize=(9,6))
    plt.plot(df_plot["in_the_moneyness"], df_plot["finite_diff_rel_error"], label="Finite Difference")
    plt.plot(df_plot["in_the_moneyness"], df_plot["montecarlo_rel_error"], label="Monte Carlo")
    plt.xlabel("Moneyness: S/K")
    plt.ylabel("Relative error")
    plt.title(f"Pricing Errors vs Moneyness for the expiry: {expiry}")
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize(lst, invert=False, eps=1e-12):
    """
    normalize a list of data to be in between 0,1, invert is to order them
    Parameters:
    ----------
    lst : list of data
    invert : decide the order
    eps : parameter to avoid division by zero
    """
    lst = np.asarray(lst, dtype=float)

    mask = np.isfinite(lst)
    out = np.full_like(lst, np.nan, dtype=float)

    if mask.sum() < 2:
        return out  # not enough valid data

    x = lst[mask]
    x = np.clip(x, eps, None)

    xmin = x.min()
    xmax = x.max()

    if np.isclose(xmax, xmin):
        out[mask] = 1.0  
    else:
        x_norm = (x - xmin) / (xmax - xmin + eps)
        out[mask] = 1 - x_norm if invert else x_norm

    return out



# 0 not brownian, 1 brownian
def build_table(f1, f2, f3, expiry_dates):
    """
    build a table for all our statistical test with values normalized to 1
    Parameters:
    ----------
    f1 : list of data for Gaussianity
    f2 : list of data for variance scaling
    f3 : list of data for kurtosis excess
    expiry_dates: list of dated analyzed
    """
    return pd.DataFrame(
        {
            "Shapiro p-value": normalize(f1),
            "Variance scaling": normalize(f2, invert=True),
            "Kurtosis": normalize(f3, invert=True)
        },
        index=pd.to_datetime(expiry_dates)
    ).sort_index()

def plot_heatmap(f1, f2, f3, expiry_dates, best_expiry, worst_expiry):
    
    heatmap_data = build_table(f1, f2, f3, expiry_dates)
    fig, ax = plt.subplots(figsize=(12,4))
    
    best_expiry = pd.to_datetime(best_expiry)
    best_index = heatmap_data.index.get_loc(best_expiry)
    worst_expiry = pd.to_datetime(worst_expiry)
    worst_index = heatmap_data.index.get_loc(worst_expiry)
    
    im = ax.imshow(
        heatmap_data.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest"
    )

    ax.set_yticks(range(len(heatmap_data.columns)))
    ax.set_yticklabels(heatmap_data.columns)
    
    ax.axvline(x=best_index,color="red",linestyle="--",linewidth=2,label="Best expiry")
    ax.annotate(best_expiry.strftime("%Y-%m-%d"),xy=(best_index, 1.02),xycoords=("data", "axes fraction"),ha="center",
    va="bottom",
    color="red",
    fontsize=10,
    fontweight="bold",
    rotation=0
    )
    ax.axvline(x=worst_index, color="orange", linestyle="--", linewidth=2,label="Worst expiry")
    ax.annotate(worst_expiry.strftime("%Y-%m-%d"),xy=(worst_index, 1.02),xycoords=("data", "axes fraction"),ha="center",
    va="bottom",
    color="orange",
    fontsize=10,
    fontweight="bold",
    rotation=0
    )
    plt.legend()
    
    n = len(heatmap_data.index)
    step = max(1, n // 12)   # circa 12 etichette al massimo

    xticks = range(0, n, step)

    ax.set_xticks(xticks)
    ax.set_xticklabels(
    heatmap_data.index[xticks].strftime("%Y-%m"),
    rotation=45,
    ha="right"
    )
    ax.set_title("Consistency of filters vs expiry dates", pad=20 )
    plt.colorbar(im, ax=ax, label="Normalized score (1 = closer to BS)")

    plt.tight_layout()
    plt.show()
