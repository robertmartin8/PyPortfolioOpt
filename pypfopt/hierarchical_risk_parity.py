"""
The ``hierarchical_risk_parity`` module implements the HRP portfolio from Marcos Lopez de Prado.
Call ``hrp_portfolio(returns)`` to generate portfolio weights.
"""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

# This code has been reproduced(with modification) from the paper:
# López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample.
# The Journal of Portfolio Management, 42(4), 59–69. https: // doi.org / 10.3905 / jpm.2016.42.4.059


def _get_cluster_var(cov, cluster_items):
    # Compute variance per cluster
    cov_slice = cov.loc[cluster_items, cluster_items]
    weights = 1 / np.diag(cov_slice)  # Inverse variance weights
    weights /= weights.sum()
    w = weights.reshape(-1, 1)
    cluster_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return cluster_var


def _get_quasi_diag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, -1]  # number of original items
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = sort_ix.append(df0)  # item 2
        sort_ix = sort_ix.sort_index()  # re-sort
        sort_ix.index = range(sort_ix.shape[0])  # re-index

    return sort_ix.tolist()


def _raw_hrp_allocation(cov, ordered_tickers):
    # Compute HRP allocation
    w = pd.Series(1, index=ordered_tickers)
    cluster_items = [ordered_tickers]  # initialize all items in one cluster

    while len(cluster_items) > 0:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        # For each pair, optimise locally.
        for i in range(0, len(cluster_items), 2):
            first_cluster = cluster_items[i]
            second_cluster = cluster_items[i + 1]
            # Form the inverse variance portfolio for this pair
            first_variance = _get_cluster_var(cov, first_cluster)
            second_variance = _get_cluster_var(cov, second_cluster)
            alpha = 1 - first_variance / (first_variance + second_variance)
            w[first_cluster] *= alpha  # weight 1
            w[second_cluster] *= 1 - alpha  # weight 2
    return w


def hrp_portfolio(returns):
    """
    Construct a hierarchical risk parity portfolio

    :param returns: asset historical returns
    :type returns: pd.DataFrame
    :return: weights for the HRP portfolio
    :rtype: dict
    :raises TypeError: if ``returns`` is not a dataframe
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns are not a dataframe")
    corr, cov = returns.corr(), returns.cov()

    # Compute distance matrix, with ClusterWarning fix as
    # per https://stackoverflow.com/questions/18952587/
    dist = ssd.squareform(((1 - corr) / 2) ** 0.5)

    link = sch.linkage(dist, "single")
    sort_ix = _get_quasi_diag(link)
    ordered_tickers = corr.index[sort_ix].tolist()
    hrp = _raw_hrp_allocation(cov, ordered_tickers)
    return dict(hrp.sort_index())
