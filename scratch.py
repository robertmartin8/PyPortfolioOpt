from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov
from pypfopt.expected_returns import mean_historical_return
from pypfopt.tests.utilities_for_tests import setup_efficient_frontier
import pandas as pd


df = pd.read_csv("pypfopt/tests/stock_returns.csv", parse_dates=True, index_col="date")
e_ret = mean_historical_return(df)
cov = sample_cov(df)


ef = setup_efficient_frontier()
w = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
"""
Volatility: 0.21671629525656422
Expected return: 0.33035542211545876
Sharpe: 1.4320816150351678
"""

ef = EfficientFrontier(e_ret, cov, weight_bounds=(0, 0.15))
w = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
"""
Volatility: 0.21671629525656422
Expected return: 0.33035542211545876
Sharpe: 1.4320816150351678
"""

ef = setup_efficient_frontier()
w = ef.min_volatility()
ef.portfolio_performance(verbose=True)
"""
Expected return: 0.1793245141665063
Volatility: 0.15915107045094778
Sharpe: 0.9981835740658117
"""

ef = setup_efficient_frontier()
w = ef.efficient_risk(0.19)
ef.portfolio_performance(verbose=True)
"""
Expected return: 0.28577470210889416
Volatility: 0.1900001239293301
Sharpe: 1.3964928761303517
"""


ef = setup_efficient_frontier()
w = ef.efficient_return(0.25)
ef.portfolio_performance(verbose=True)
"""
Expected return: 0.2500000000006342
Volatility: 0.17388540121530308
Sharpe: 1.3205072040538786
"""

ef = EfficientFrontier(e_ret, cov)
sharpes = []
for i in range(10):
    ef.max_sharpe(risk_free_rate=i / 100)
    sharpe = ef.portfolio_performance(verbose=True)[2]
    sharpes.append(sharpe)

ef = setup_efficient_frontier()
w = ef.max_sharpe(alpha=1)
sum(ef.weights > 0.02)
ef.portfolio_performance(verbose=True)


ef = setup_efficient_frontier()
w = ef.min_volatility(alpha=1)
sum(ef.weights > 0.02)
ef.portfolio_performance(verbose=True)
"""
Expected return: 0.2211888419683154
Volatility: 0.18050174016287326
Sharpe: 1.1133499289183508
"""


# test shorts
e_ret[::2] *= -1
ef = EfficientFrontier(e_ret, cov, weight_bounds=(None, None))
ef.max_sharpe()


# market neutral
ef = setup_efficient_frontier()
ef.bounds = ((-1, 1),) * 20
ef.max_sharpe(market_neutral=True)
