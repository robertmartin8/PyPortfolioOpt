"""
The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimisation – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from . import base_optimizer


class HRPOpt(base_optimizer.BaseOptimizer):
    """
    A HRPOpt object (inheriting from BaseOptimizer) constructs a hierarchical
    risk parity portfolio.

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.Series

    - Output:

        - ``weights`` - np.ndarray
        - ``clusters`` - linkage matrix corresponding to clustered assets.

    Public methods:

    - ``optimize()`` calculates weights using HRP
    - ``plot_dendrogram()`` plots the clusters
    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimised portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, returns):
        """
        :param returns: asset historical returns
        :type returns: pd.DataFrame
        :raises TypeError: if ``returns`` is not a dataframe
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.clusters = None
        tickers = list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """
        Compute the variance per cluster

        :param cov: covariance matrix
        :type cov: np.ndarray
        :param cluster_items: tickers in the cluster
        :type cluster_items: list
        :return: the variance per cluster
        :rtype: float
        """
        # Compute variance per cluster
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)  # Inverse variance weights
        weights /= weights.sum()
        w = weights.reshape(-1, 1)
        cluster_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return cluster_var

    @staticmethod
    def _get_quasi_diag(link):
        """
        Sort clustered items by distance

        :param link: linkage matrix after clustering
        :type link: np.ndarray
        :return: sorted list of tickers
        :rtype: list
        """
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

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """
        Given the clusters, compute the portfolio that minimises risk.

        :param cov: covariance matrix
        :type cov: np.ndarray
        :param ordered_tickers: list of tickers ordered by distance
        :type ordered_tickers: str list
        :return: raw portfolio weights
        :rtype: pd.Series
        """
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
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha  # weight 1
                w[second_cluster] *= 1 - alpha  # weight 2
        return w

    def optimize(self):
        """
        Construct a hierarchical risk parity portfolio

        :return: weights for the HRP portfolio
        :rtype: dict
        """
        corr, cov = self.returns.corr(), self.returns.cov()

        # Compute distance matrix, with ClusterWarning fix as
        # per https://stackoverflow.com/questions/18952587/
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)

        self.clusters = sch.linkage(dist, "single")
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        weights = dict(hrp.sort_index())
        self.set_weights(weights)
        return weights

    def plot_dendrogram(self, show_tickers=True, filename=None, showfig=True):
        """
        Plot the clusters in the form of a dendrogram.

        :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                            defaults to True
        :type show_tickers: bool, optional
        :param filename: name of the file to save to, defaults to None (doesn't save)
        :type filename: str, optional
        :param showfig: whether to plt.show() the figure, defaults to True
        :type showfig: bool, optional
        :raises ImportError: if matplotlib is not installed
        :return: matplotlib axis
        :rtype: matplotlib.axes object
        """

        try:
            import matplotlib.pyplot as plt
        except (ModuleNotFoundError, ImportError):
            raise ImportError("Please install matplotlib via pip or poetry")

        if self.clusters is None:
            self.optimize()

        fig, ax = plt.subplots()
        if show_tickers:
            sch.dendrogram(self.clusters, labels=self.tickers, ax=ax)
            plt.xticks(rotation=90)
            plt.tight_layout()
        else:
            sch.dendrogram(self.clusters, ax=ax)

        if filename:
            plt.savefig(fname=filename, dpi=300)

        if showfig:
            plt.show()

        return ax

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02, frequency=252):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio
        assuming returns are daily

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :param frequency: number of time periods in a year, defaults to 252 (the number
                            of trading days in a year)
        :type frequency: int, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        return base_optimizer.portfolio_performance(
            self.weights,
            self.returns.mean() * frequency,
            self.returns.cov() * frequency,
            verbose,
            risk_free_rate,
        )
