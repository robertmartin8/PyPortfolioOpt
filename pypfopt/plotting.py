"""
The ``plotting`` module houses all the functions to generate various plots.

Currently implemented:

  - ``plot_covariance`` - plot a correlation matrix
  - ``plot_dendrogram`` - plot the hierarchical clusters in a portfolio
  - ``plot_efficient_frontier`` – plot the efficient frontier, using the CLA algorithm.
  - ``plot_weights`` - bar chart of weights
"""

import numpy as np
from . import risk_models
import scipy.cluster.hierarchy as sch

try:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-deep")
except (ModuleNotFoundError, ImportError):
    raise ImportError("Please install matplotlib via pip or poetry")


def _plot_io(**kwargs):
    """
    Helper method to optionally save the figure to file.

    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param dpi: dpi of figure to save or plot, defaults to 300
    :type dpi: int (between 50-500)
    :param showfig: whether to plt.show() the figure, defaults to True
    :type showfig: bool, optional
    """
    filename = kwargs.get("filename", None)
    showfig = kwargs.get("showfig", True)
    dpi = kwargs.get("dpi", 300)

    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:
        plt.show()


def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax


def plot_dendrogram(hrp, show_tickers=True, **kwargs):
    """
    Plot the clusters in the form of a dendrogram.

    :param hrp: HRPpt object that has already been optimized.
    :type hrp: object
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to True
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if hrp.clusters is None:
        hrp.optimize()

    fig, ax = plt.subplots()
    if show_tickers:
        sch.dendrogram(hrp.clusters, labels=hrp.tickers, ax=ax, orientation="top")
        plt.xticks(rotation=90)
        plt.tight_layout()
    else:
        sch.dendrogram(hrp.clusters, no_labels=True, ax=ax)

    _plot_io(**kwargs)

    return ax


def plot_efficient_frontier(cla, points=100, show_assets=True, **kwargs):
    """
    Plot the efficient frontier based on a CLA object

    :param points: number of points to plot, defaults to 100
    :type points: int, optional
    :param show_assets: whether we should plot the asset risks/returns also, defaults to True
    :type show_assets: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to True
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if cla.weights is None:
        cla.max_sharpe()
    optimal_ret, optimal_risk, _ = cla.portfolio_performance()

    if cla.frontier_values is None:
        cla.efficient_frontier(points=points)

    mus, sigmas, _ = cla.frontier_values

    fig, ax = plt.subplots()
    ax.plot(sigmas, mus, label="Efficient frontier")

    if show_assets:
        ax.scatter(
            np.sqrt(np.diag(cla.cov_matrix)),
            cla.expected_returns,
            s=30,
            color="k",
            label="assets",
        )

    ax.scatter(optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal")
    ax.legend()
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")

    _plot_io(**kwargs)
    return ax


def plot_weights(weights, **kwargs):
    """
    Plot the portfolio weights as a horizontal bar chart

    :param weights: the weights outputted by any PyPortfolioOpt optimiser
    :type weights: {ticker: weight} dict
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    return ax
