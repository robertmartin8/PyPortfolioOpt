"""
The ``risk_models`` module provides functions for estimating the covariance matrix given
historical returns. Because of the complexity of estimating covariance matrices
(and the importance of efficient computations), this module mostly provides a convenient
wrapper around the underrated `sklearn.covariance` module.

The format of the data input is the same as that in :ref:`expected-returns`.

**Currently implemented:**

- sample covariance
- semicovariance
- exponentially weighted covariance
- mininum covariance determinant
- shrunk covariance matrices:

    - manual shrinkage
    - Ledoit Wolf shrinkage
    - Oracle Approximating shrinkage
"""
import warnings
import numpy as np
import pandas as pd
from sklearn import covariance
from .expected_returns import daily_price_returns


def sample_cov(prices, frequency=252):
    """
    Calculate the annualised sample covariance matrix of (daily) asset returns.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: annualised sample covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = daily_price_returns(prices)
    return daily_returns.cov() * frequency


def semicovariance(prices, benchmark=0, frequency=252):
    """
    Estimate the semicovariance matrix, i.e the covariance given that
    the returns are less than the benchmark.

    .. semicov = E([min(r_i - B, 0)] . [min(r_j - B, 0)])

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param benchmark: the benchmark return, defaults to 0.
    :type benchmark: float
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: semicovariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = daily_price_returns(prices)
    drops = np.fmin(daily_returns - benchmark, 0)
    return drops.cov() * frequency


def _pair_exp_cov(X, Y, span=180):
    """
    Calculate the exponential covariance between two timeseries of returns.

    :param X: first time series of returns
    :type X: pd.Series
    :param Y: second time series of returns
    :type Y: pd.Series
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :return: the exponential covariance between X and Y
    :rtype: float
    """
    covariation = (X - X.mean()) * (Y - Y.mean())
    # Exponentially weight the covariation and take the mean
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    return covariation.ewm(span=span).mean()[-1]


def exp_cov(prices, span=180, frequency=252):
    """
    Estimate the exponentially-weighted covariance matrix, which gives
    greater weight to more recent data.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: annualised estimate of exponential covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    daily_returns = daily_price_returns(prices)
    N = len(assets)

    # Loop over matrix, filling entries with the pairwise exp cov
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(
                daily_returns.iloc[:, i], daily_returns.iloc[:, j], span
            )
    return pd.DataFrame(S * frequency, columns=assets, index=assets)


def min_cov_determinant(prices, frequency=252, random_state=None):
    """
    Calculate the minimum covariance determinant, an estimator of the covariance matrix
    that is more robust to noise.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param random_state: random seed to make results reproducible, defaults to None
    :type random_state: int, optional
    :return: annualised estimate of covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    X = prices.pct_change().dropna(how="all")
    X = np.nan_to_num(X.values)
    raw_cov_array = covariance.fast_mcd(X, random_state=random_state)[1]
    return pd.DataFrame(raw_cov_array, index=assets, columns=assets) * frequency


class CovarianceShrinkage:
    """
    Provide methods for computing shrinkage estimates of the covariance matrix, using the
    sample covariance matrix and choosing the structured estimator to be an identity matrix
    multiplied by the average sample variance. The shrinkage constant can be input manually,
    though there exist methods (notably Ledoit Wolf) to estimate the optimal value.

    Instance variables:

    - ``X`` (returns)
    - ``S`` (sample covariance matrix)
    - ``delta`` (shrinkage constant)
    """

    def __init__(self, prices, frequency=252):
        """
        :param prices: adjusted closing prices of the asset, each row is a date and each column is a ticker/id.
        :type prices: pd.DataFrame
        :param frequency: number of time periods in a year, defaults to 252 (the number of trading days in a year)
        :type frequency: int, optional
        """
        if not isinstance(prices, pd.DataFrame):
            warnings.warn("prices are not in a dataframe", RuntimeWarning)
            prices = pd.DataFrame(prices)
        self.frequency = frequency
        self.X = prices.pct_change().dropna(how="all")
        self.S = self.X.cov().values
        self.delta = None  # shrinkage constant

    def format_and_annualise(self, raw_cov_array):
        """
        Helper method which annualises the output of shrinkage calculations,
        and formats the result into a dataframe

        :param raw_cov_array: raw covariance matrix of daily returns
        :type raw_cov_array: np.ndarray
        :return: annualised covariance matrix
        :rtype: pd.DataFrame
        """
        assets = self.X.columns
        return (
            pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        )

    def shrunk_covariance(self, delta=0.2):
        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.

        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        self.delta = delta
        N = self.S.shape[1]
        # Shrinkage target
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self.format_and_annualise(shrunk_cov)

    def ledoit_wolf(self):
        """
        Calculate the Ledoit-Wolf shrinkage estimate.

        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = covariance.ledoit_wolf(X)
        return self.format_and_annualise(shrunk_cov)

    def oracle_approximating(self):
        """
        Calculate the Oracle Approximating Shrinkage estimate

        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        X = np.nan_to_num(self.X.values)
        shrunk_cov, self.delta = covariance.oas(X)
        return self.format_and_annualise(shrunk_cov)


class ConstantCorrelation(CovarianceShrinkage):
    """
    Shrinks towards constant correlation matrix
    if shrink is specified, then this constant is used for shrinkage

    The notation follows Ledoit and Wolf (2003, 2004) version 04/2014
    """

    def shrink(self):
        """
        Calculate the Constant-Correlation covariance matrix.

        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        x = np.nan_to_num(self.X.values)

        # de-mean returns
        t, n = np.shape(x)
        meanx = x.mean(axis=0)
        x = x - np.tile(meanx, (t, 1))

        # compute sample covariance matrix
        sample = (1.0 / t) * np.dot(x.T, x)

        # compute prior
        var = np.diag(sample).reshape(-1, 1)
        sqrtvar = np.sqrt(var)
        _var = np.tile(var, (n,))
        _sqrtvar = np.tile(sqrtvar, (n,))
        r_bar = (np.sum(sample / (_sqrtvar * _sqrtvar.T)) - n) / (n * (n - 1))
        prior = r_bar * (_sqrtvar * _sqrtvar.T)
        prior[np.eye(n) == 1] = var.reshape(-1)

        # compute shrinkage parameters and constant
        if self.delta is None:

            # what we call pi-hat
            y = x ** 2.0
            phi_mat = np.dot(y.T, y) / t - 2 * np.dot(x.T, x) * sample / t + sample ** 2
            phi = np.sum(phi_mat)

            # what we call rho-hat
            term1 = np.dot((x ** 3).T, x) / t
            help_ = np.dot(x.T, x) / t
            help_diag = np.diag(help_)
            term2 = np.tile(help_diag, (n, 1)).T * sample
            term3 = help_ * _var
            term4 = _var * sample
            theta_mat = term1 - term2 - term3 + term4
            theta_mat[np.eye(n) == 1] = np.zeros(n)
            rho = sum(np.diag(phi_mat)) + r_bar * np.sum(
                np.dot((1.0 / sqrtvar), sqrtvar.T) * theta_mat
            )

            # what we call gamma-hat
            gamma = np.linalg.norm(sample - prior, "fro") ** 2

            # compute shrinkage constant
            kappa = (phi - rho) / gamma
            shrinkage = max(0.0, min(1.0, kappa / t))
            self.delta = shrinkage
        else:
            # use specified constant
            shrinkage = self.delta

        # compute the estimator
        sigma = shrinkage * prior + (1 - shrinkage) * sample
        return self.format_and_annualise(sigma)


class SingleIndex(CovarianceShrinkage):
    """
    This estimator is a weighted average of the sample
    covariance matrix and a "prior" or "shrinkage target".
    Here, the prior is given by a one-factor model.
    The factor is equal to the cross-sectional average
    of all the random variables.

    The notation follows Ledoit and Wolf (2003), version: 04/2014

    """

    def shrink(self):
        """
        Calculate the Constant-Correlation covariance matrix.

        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        x = np.nan_to_num(self.X.values)

        # de-mean returns
        t, n = np.shape(x)
        meanx = x.mean(axis=0)
        x = x - np.tile(meanx, (t, 1))
        xmkt = x.mean(axis=1).reshape(t, 1)

        # compute sample covariance matrix
        sample = np.cov(np.append(x, xmkt, axis=1), rowvar=False) * (t - 1) / t
        covmkt = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        prior = np.dot(covmkt, covmkt.T) / varmkt
        prior[np.eye(n) == 1] = np.diag(sample)

        # compute shrinkage parameters
        if self.delta is None:
            c = np.linalg.norm(sample - prior, "fro") ** 2
            y = x ** 2
            p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample ** 2)
            # r is divided into diagonal
            # and off-diagonal terms, and the off-diagonal term
            # is itself divided into smaller terms
            rdiag = 1 / t * np.sum(y ** 2) - sum(np.diag(sample) ** 2)
            z = x * np.tile(xmkt, (n,))
            v1 = 1 / t * np.dot(y.T, z) - np.tile(covmkt, (n,)) * sample
            roff1 = (
                np.sum(v1 * np.tile(covmkt, (n,)).T) / varmkt
                - np.sum(np.diag(v1) * covmkt.T) / varmkt
            )
            v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
            roff3 = (
                np.sum(v3 * np.dot(covmkt, covmkt.T)) / varmkt ** 2
                - np.sum(np.diag(v3).reshape(-1, 1) * covmkt ** 2) / varmkt ** 2
            )
            roff = 2 * roff1 - roff3
            r = rdiag + roff

            # compute shrinkage constant
            k = (p - r) / c
            shrinkage = max(0, min(1, k / t))
            self.delta = shrinkage
        else:
            # use specified constant
            shrinkage = self.delta

        # compute the estimator
        sigma = shrinkage * prior + (1 - shrinkage) * sample
        return self.format_and_annualise(sigma)
