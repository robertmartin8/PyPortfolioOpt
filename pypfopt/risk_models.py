"""
This module implements possible models for risk of a portfolio
"""
import pandas as pd
import numpy as np
import warnings


def sample_cov(prices):
    """
    Calculates the sample covariance matrix of daily returns, then annualises.
    :param daily_returns: Daily returns, each row is a date and each column is a ticker
    :type daily_returns: pd.DataFrame or array-like
    :returns: annualised sample covariance matrix of daily returns
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    daily_returns = prices.pct_change().dropna(how="all")

    return daily_returns.cov() * 252


class LedoitWolfShrinkage:
    """
    Originally described in Ledoit and Wolf (2001).

    https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/single_index_covariance_estimator.py
    
    This is 
    """
    # TODO implement a default identity shrinkage

    def __init__(self, daily_returns):
        self.X = daily_returns
        self.F = None  # shrinkage target
        self.delta = None  # shrinkage constant
        self.S = daily_returns.cov()  # sample cov is the MLE

    def identity_shrinkage_target(self):
        #  TODO this requires default rho
        pass

    def _pi_matrix(self):
        """
        Estimates the asymptotic variances of the scaled entries of the sample covariance matrix.
        This is used to calculate pi and rho.
        :return: scaled asymptotic variances
        :rtype: NxN np.array of, where N is the number of assets.
        """
        # TODO optimize this?
        T, N = self.X.shape
        Xc = self.X - self.X.mean()
        pi_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                pi_matrix[i, j] = pi_matrix[j, i] = np.sum(
                    (Xc.iloc[:, i] * Xc.iloc[:, j] - self.S.iloc[i, j]) ** 2
                )
        return pi_matrix / T

    def _pi(self):
        """
        Estimates the sum of asymptotic variances of the scaled entries of the
        sample covariance matrix. One of the three components of the optimal shrinkage
        constant
        :return: pi(hat)
        :rtype: float
        """
        pi_matrix = self._pi_matrix()
        return np.sum(pi_matrix)

    def _gamma(self):
        """
        Gamma estimates deviance of the shrinkage target (i.e sum of element-wise squared distances
        between shrinkage target matrix and sample covariance matrix).
        :return: gamma
        :rtype: float
        """
        return np.sum((self.F - self.S) ** 2)

    def _optimal_shrinkage_constant(self):
        """
        Calculates the optimal value of the shrinkage constant, as per Ledoit and Wolf. This
        is truncated to [0, 1], in the rare case that k/T is outside of this range.
        :return: optimal shrinkage constant
        :rtype: float between 0 and 1.
        """
        T = self.X.shape[0]
        pi = self._pi()
        rho = self._rho()
        gamma = self._gamma()
        kappa = (pi - rho) / gamma
        self.delta = max(0, min(kappa / T, 1))
        return self.delta

    # TODO rename this
    def shrink(self):
        if self.delta is None:
            self._optimal_delta()

        # Return shrunk matrix
        S_hat = self.delta * self.F + (1 - self.delta) * self.S
        return S_hat


class SingleFactorShrinkage(LedoitWolfShrinkage):
    """
    See Sharpe, 1963
    """

    def single_factor_shrinkage_target(self):
        # estimate beta from CAPM (use precomputed sample covariance to calculate beta)
        # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line
        var_market = self.S[0, 0]
        y = self.X[:, 0]
        beta = self.S[0, :] / var_market
        alpha = np.mean(self.X, 0) - beta * np.mean(y)

        # get residuals and their variance
        eps = self.X - alpha - np.matrix(y).T * np.matrix(beta)
        D = np.diag(np.var(eps, 0))

        self.F = var_market * np.matrix(beta).T * np.matrix(beta) + D
        return self.F


class ConstantCorrelationShrinkage(LedoitWolfShrinkage):
    """
    Shrinkage target is constant correlation, as per Ledoit and Wolf (2004).
    """

    # TODO implement
    def _rho(self):
        Xc = self.X - self.X.mean()
        T, N = self.X.shape

        pi_diag = np.trace(self._pi_matrix())

        return np.sum(R)

    def constant_correlation_shrinkage_target(self):
        """
        Calculates the constant correlation shrinkage target, as per Ledoit and Wolf 2004.
        All pairwise correlations are assumed to be equal to the mean of all correlations,
        and the diagonal is just the std of each asset's returns.
        Note that these functions particularly depend on X being an pandas dataframe, because
        it provides functions that are very robust to missing data.
        :return: constant correlation shrinkage target
        :rtype: np.array
        """
        correlation_matrix = self.X.corr()
        r_bar = np.mean(correlation_matrix.values)  # average correlation
        # Use numpy's broadcasting to calculate s_ii * s_jj
        # Then f_ij = r_bar * sqrt(s_ii * s_jj)
        F = r_bar * np.sqrt(self.X.std().values * self.X.std().values.reshape((-1, 1)))
        # f_ii = s_ii
        np.fill_diagonal(F, F.diagonal() / r_bar)
        self.F = F
        return self.F
