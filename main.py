import yfinance as yf
import pandas as pd
import config

from src.get_data import (
    get_bday_calendar,
    get_expiries_options,
    get_trading_days,
    get_standardized_log_returns,
    get_underlying_price,
)

from src.filtering_data import (
    get_shap_filter,
    get_filter_variance,
    get_kurtosis_filter,
    get_best_expiry,
    get_worst_expiry
) 

from src.compute_parameters_options import(
    compute_implied_vol   
)

from src.numerics_pricing import(
    fd_with_error,
    montercarlo_df   
)

from src.print_data import(
    print_results
)

from src.get_plots import(
    plot_iv_smile,
    plot_pricing_errors,
    get_qq_plot,
    variance_vs_time,
    plot_pricing_comparison,
    kurtosis_vs_days,
    pvalue_vs_days,
    plot_heatmap
)

from src.option_parameters import(
    option_data
)

def main():
    
    #0. setup
    
    ticker_opt = yf.Ticker(config.ticker_symbol)
    my_calendar = get_bday_calendar()
    
    #1. I choose the option ticker, download the options, fix the calendar, look at their expiries

    expiries = get_expiries_options(ticker_opt)
    days_to_expiry = get_trading_days(expiries, my_calendar)
    
    #2. Download data of the corresponding underlying

    underlying_log_ret_std = get_standardized_log_returns(ticker_opt)
    S = get_underlying_price(ticker_opt)
    
    #3.Construct statystical filters, select the best and worst expiry for Black-Scholes
    
    shapiro_test = get_shap_filter(underlying_log_ret_std, days_to_expiry)
    variance_test = get_filter_variance(underlying_log_ret_std, days_to_expiry)
    kurtosis_test = get_kurtosis_filter(underlying_log_ret_std, days_to_expiry )

    best_expiry, best_pvalue, best_variance, best_kurtosis = get_best_expiry(kurtosis_test, shapiro_test,
                                                                             variance_test, expiries)
    days_to_best_expiry = get_trading_days([best_expiry], my_calendar)[0]

    worst_expiry, worst_pvalue, worst_variance, worst_kurtosis = get_worst_expiry(kurtosis_test, shapiro_test,
                                                                                  variance_test, expiries)
    days_to_worst_expiry = get_trading_days([worst_expiry], my_calendar)[0]

    #4. Download option data and relevant paremeters for call and put options

    best_option, T_best, r_eff_best  = option_data(ticker_opt, best_expiry, S) #closest option to BS
    
    worst_option, T_worst, r_eff_worst = option_data(ticker_opt, worst_expiry, S) #most far option from BS
    
    #5. Compute implied volatility
    
    best_df_iv = compute_implied_vol(best_option, S, T_best, r_eff_best)
    worst_df_iv = compute_implied_vol(worst_option, S, T_worst, r_eff_worst)
    
    #6. BS pricing with finite difference method
    
    best_df_fd = fd_with_error(S, best_df_iv, T_best, r_eff_best, config.t_steps, config.price_steps)
    worst_df_fd = fd_with_error(S, worst_df_iv, T_worst, r_eff_worst, config.t_steps, config.price_steps)

    #7. BS pricing with Montecarlo
    
    best_df_priced = montercarlo_df(best_df_fd, S, T_best, r_eff_best, config.t_mc, config.num_simulations)
    worst_df_priced = montercarlo_df(worst_df_fd, S, T_worst, r_eff_worst, config.t_mc, config.num_simulations)
    
    #8. Plots of statistical tests, implied volatility smile/skew, relative errors in pricing
 
    pvalue_vs_days(expiries, shapiro_test, best_expiry, worst_expiry,
                "p-value Shapiro vs expiry date", "Days to expiry (log scale)", "p-value")
    variance_vs_time(underlying_log_ret_std, days_to_expiry, days_to_best_expiry, days_to_worst_expiry)
    kurtosis_vs_days(expiries, kurtosis_test, best_expiry, worst_expiry,
                "Kurtosis vs expiry date", "days_to_expiry", "kurtosis")
    plot_heatmap(shapiro_test, variance_test, kurtosis_test, expiries, best_expiry, worst_expiry)
    
    get_qq_plot(underlying_log_ret_std, days_to_best_expiry, "best expiry")
    get_qq_plot(underlying_log_ret_std, days_to_worst_expiry, "worst expiry")
    
    plot_iv_smile(best_df_iv, best_expiry)
    plot_iv_smile(worst_df_iv, worst_expiry)
    
    plot_pricing_comparison(best_df_priced, best_expiry)
    plot_pricing_errors(best_df_priced, best_expiry)
    plot_pricing_comparison(worst_df_priced, worst_expiry)
    plot_pricing_errors(worst_df_priced, worst_expiry)
    
    #9. Prints the results
    
    print_results(best_df_priced, best_expiry, best_pvalue, best_variance, best_kurtosis)
    print_results(worst_df_priced, worst_expiry, worst_pvalue, worst_variance, worst_kurtosis)

if __name__ == "__main__":
    main()
