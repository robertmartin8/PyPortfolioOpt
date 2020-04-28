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
