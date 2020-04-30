import os

import pandas as pd
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA


def resource(name):
    return os.path.join(os.path.dirname(__file__), "resources", name)


def get_data():
    return pd.read_csv(resource("stock_prices.csv"), parse_dates=True, index_col="date")


def get_benchmark_data():
    return pd.read_csv(resource("spy_prices.csv"), parse_dates=True, index_col="date")


def get_market_caps():
    mcaps = {
        "GOOG": 927e9,
        "AAPL": 1.19e12,
        "FB": 574e9,
        "BABA": 533e9,
        "AMZN": 867e9,
        "GE": 96e9,
        "AMD": 43e9,
        "WMT": 339e9,
        "BAC": 301e9,
        "GM": 51e9,
        "T": 61e9,
        "UAA": 78e9,
        "SHLD": 0,
        "XOM": 295e9,
        "RRC": 1e9,
        "BBY": 22e9,
        "MA": 288e9,
        "PFE": 212e9,
        "JPM": 422e9,
        "SBUX": 102e9,
    }
    return mcaps


def setup_efficient_frontier(data_only=False):
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return mean_return, sample_cov_matrix
    return EfficientFrontier(mean_return, sample_cov_matrix)


def setup_cla(data_only=False):
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return mean_return, sample_cov_matrix
    return CLA(mean_return, sample_cov_matrix)
