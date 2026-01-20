import pandas as pd

"""
Print summary of the results for the option: 
-date
-statistical data  
- basic statistic on implied volatility, pricing errors with FD and Monte Carlo
"""

def print_results(df,expiry, pvalue, variance, kurtosis):
    """
    It prints a summary of the results
    Parameters:
    ----------
    df:  dataframe with all the results
    expiry: expiry date of the option
    pvalue : pvalue test of the underlying
    variance : variance scaling test of the underlying
    kurtosis : variance scaling test of the underlying
    """
    if pvalue > 0.5:
        print("*" * 55)
        print(f"The best expiry is {expiry}")
        print(f"The coresponding Shapiro p-value is {pvalue}")
        print(f"The coresponding variance is {variance}")
        print(f"The coresponding kurtosis excess is {kurtosis}")
        print("*" * 55)
    else:
        print("*" * 55)
        print(f"The worst expiry is: {expiry}")
        print(f"The coresponding Shapiro p-value is:  {pvalue}")
        print(f"The coresponding scaling variance is: {variance}")
        print(f"The coresponding kurtosis excess is:  {kurtosis}")
        print("*" * 55)
    
    df = df.copy()
    df_print = df[["impl_vol", "finite_diff_rel_error","montecarlo_rel_error"]].rename(columns=
    {"impl_vol": "Implied Vol",
     "finite_diff_rel_error": "Finite Difference rel error",
     "montecarlo_rel_error": "Monte Carlo rel error",})
    summary = df_print.describe().loc[["min", "mean", "std", "50%", "max"]]
    summary.rename(index={"50%":"median"}, inplace=True)
    print(summary)