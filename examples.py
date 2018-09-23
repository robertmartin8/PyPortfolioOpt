import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.hierarchical_risk_parity import hrp_portfolio
from pypfopt.value_at_risk import CVAROpt
from pypfopt import discrete_allocation


# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")
returns = df.pct_change().dropna(how="all")
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Long-only Maximum Sharpe portfolio, with discretised weights
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
latest_prices = discrete_allocation.get_latest_prices(df)
allocation, leftover = discrete_allocation.portfolio(weights, latest_prices)
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

"""
Expected annual return: 33.0%
Annual volatility: 21.7%
Sharpe Ratio: 1.43

Discrete allocation: {'MA': 14, 'FB': 12, 'PFE': 51, 'BABA': 5, 'AAPL': 5,
                      'AMZN': 0, 'BBY': 9, 'SBUX': 6, 'GOOG': 1}
Funds remaining: $12.15
"""

# Long-only minimum volatility portfolio, with a weight cap and regularisation
# e.g if we want at least 15/20 tickers to have non-neglible weights, and no
# asset should have a weight greater than 10%
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.10), gamma=1)
weights = ef.min_volatility()
print(weights)
ef.portfolio_performance(verbose=True)

"""
{
    "GOOG": 0.07350956640872872,
    "AAPL": 0.030014017863649482,
    "FB": 0.1,
    "BABA": 0.1,
    "AMZN": 0.020555866446753328,
    "GE": 0.04052056082259943,
    "AMD": 0.00812443078787937,
    "WMT": 0.06506870608367901,
    "BAC": 0.008164561664321555,
    "GM": 0.1,
    "T": 0.06581732376444831,
    "UAA": 0.04764331094366604,
    "SHLD": 0.04233556511047908,
    "XOM": 0.06445358180591973,
    "RRC": 0.0313848213281047,
    "BBY": 0.02218378020003044,
    "MA": 0.068553464907087,
    "PFE": 0.059025401478094965,
    "JPM": 0.015529411963789761,
    "SBUX": 0.03711562842076907,
}

Expected annual return: 22.7%
Annual volatility: 12.7%
Sharpe Ratio: 1.63
"""

# A long/short portfolio maximising return for a target volatility of 10%,
# with a shrunk covariance matrix risk model
shrink = risk_models.CovarianceShrinkage(df)
S = shrink.ledoit_wolf()
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights = ef.efficient_risk(target_risk=0.10)
ef.portfolio_performance(verbose=True)

"""
Expected annual return: 29.8%
Annual volatility: 10.0%
Sharpe Ratio: 2.77
"""

# A market-neutral Markowitz portfolio finding the minimum volatility
# for a target return of 20%
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights = ef.efficient_return(target_return=0.20, market_neutral=True)
ef.portfolio_performance(verbose=True)

"""
Expected annual return: 20.0%
Annual volatility: 16.5%
Sharpe Ratio: 1.09
"""


# Custom objective
def utility_obj(weights, mu, cov_matrix, k=1):
    return -weights.dot(mu) + k * np.dot(weights.T, np.dot(cov_matrix, weights))


ef = EfficientFrontier(mu, S)
ef.custom_objective(utility_obj, ef.expected_returns, ef.cov_matrix, 1)
ef.portfolio_performance(verbose=True)

"""
Expected annual return: 40.1%
Annual volatility: 29.2%
Sharpe Ratio: 1.30
"""

ef.custom_objective(utility_obj, ef.expected_returns, ef.cov_matrix, 2)
ef.portfolio_performance(verbose=True)

"""
Expected annual return: 36.6%
Annual volatility: 24.7%
Sharpe Ratio: 1.39
"""

# CVaR optimisation
vr = CVAROpt(returns)
vr.min_cvar()
print(vr.clean_weights())

"""
{'GOOG': 0.10886,
 'AAPL': 0.0,
 'FB': 0.02598,
 'BABA': 0.57691,
 'AMZN': 0.0,
 'GE': 0.01049,
 'AMD': 0.0138,
 'WMT': 0.01581,
 'BAC': 0.01049,
 'GM': 0.03463,
 'T': 0.01049,
 'UAA': 0.07782,
 'SHLD': 0.04184,
 'XOM': 0.00931,
 'RRC': 0.0,
 'BBY': 0.01748,
 'MA': 0.03782,
 'PFE': 0.0,
 'JPM': 0.0,
 'SBUX': 0.00828}
 """


# Hierarchical risk parity
weights = hrp_portfolio(returns)
print(weights)

"""
{'AAPL': 0.022258941278778397,
 'AMD': 0.02229402179669211,
 'AMZN': 0.016086842079875,
 'BABA': 0.07963382071794091,
 'BAC': 0.014409222455552262,
 'BBY': 0.0340641943824504,
 'FB': 0.06272994714663534,
 'GE': 0.05519063444162849,
 'GM': 0.05557666024185722,
 'GOOG': 0.049560084289929286,
 'JPM': 0.017675709092515708,
 'MA': 0.03812737349732021,
 'PFE': 0.07786528342813454,
 'RRC': 0.03161528695094597,
 'SBUX': 0.039844436656239136,
 'SHLD': 0.027113184241298865,
 'T': 0.11138956508836476,
 'UAA': 0.02711590957075009,
 'WMT': 0.10569551148587905,
 'XOM': 0.11175337115721229}
"""
