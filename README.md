# Statistical Expiry Selection and Pricing within the Black–Scholes Framework
*A Python framework to select expiries satisfying the Black-Scholes assumptions and to price European options.*

## 1. Motivation
The Black-Scholes model, a milestone in **quantitative finance**, assumes that the underlying follows a Brownian motion. This assumption is sensitive to the time window of the expiry date.

Thus, we propose a **data-driven statistical approach** to select the expiry closest to the Brownian motion hypothesis. To investigate the consequences of deviations, we compare the Black-Scholes predictions against **market mid-prices**, both for the case closest to Black-Scholes and the most deviating one.

## 2. The logic of the code is the following

Selection process:
- Download underlying price data using **yfinance**.
- Perform statistical tests on log-returns: Shapiro-Wilk, kurtosis, variance scaling.
- Select the **best** and **worst** expiry according to the Black-Scholes assumptions.

Pricing process:
- Extract the implied volatility with a Newton root-finding method.
- Price with **finite differences**.
- Price with **Monte Carlo simulations**.
- Compare model and market prices.

## 3. Statistical tests

We implement the following data-driven tests to check the Brownian motion hypothesis:
- **Normality** of the log-returns with the Shapiro-Wilk test.
- **Variance scaling**: check the linear scaling of the variance with time.
- **Heavy tails**: evaluate the excess kurtosis of the distribution.

## 4. Selection process

Best/worst expiry selection:
1. Select the expiries that pass/fail both the Shapiro-Wilk and kurtosis tests.
2. Choose the one with the best/worst variance scaling test.

## 5. Volatility smile

- Calculate the implied volatility for the best and worst scenarios using a Newton root-finding method across strikes and plot the volatility smile.

## 6. Pricing process

- Price European call options using the explicit finite difference method.
- Price European call options using Monte Carlo simulation with antithetic sampling.
- Compute **relative errors** against market mid-prices.

## 7. Results

- Mid-range expiries are usually closer to the Black-Scholes hypothesis.
- Tests tend to fail for shorter and longer maturities.
- Pricing with implied volatility is relatively robust to these deviations.

## 8. Future developments

- Combine tests into a **continuous scoring** to evaluate all expiries.
- Extend pricing to **historical volatility** to compare errors across expiries and volatilities.
- Extend the analysis to the **Greeks**.

## 9. Requirements

- Python ≥ 3.9   
- numpy
- pandas
- scipy
- matplotlib
- yfinance
- pandas_market_calendars

## 10. Usage

- to install the packages: pip install -r requirements.txt
- to run the code: python main.py

***Disclaimer:***
This project is intended for educational and research purposes only.
It is not designed for live trading, portfolio management, or real-world financial decision making.
