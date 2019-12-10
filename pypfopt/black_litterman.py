"""
Module docstring

"""
import warnings
import numpy as np
import pandas as pd

from . import base_optimizer


def market_implied_prior_returns():
    pass


def market_implied_risk_aversion():
    pass


class BlackLittermanModel(base_optimizer.BaseOptimizer):

    """
    Instance variables:

    - ``n_assets``
    - ``tickers``
    - ``weights``
    """

    def __init__(
        self,
        cov_matrix,
        pi=None,
        absolute_views=None,
        Q=None,
        P=None,
        omega=None,
        tau=0.05,
    ):
        """
        :param cov_matrix: NxN covariance matrix of returns
        :type cov_matrix: pd.DataFrame or np.ndarray
        :param pi: Nx1 prior estimate of returns (e.g market-implied returns), defaults to None
        :type pi: np.ndarray, pd.Series, optional
        :param absolute_views: a colleciton of K absolute views on a subset of assets,
                               defaults to None. If this is provided, we do not need P, Q.
        :type absolute_views: pd.Series or dict, optional
        :param Q: Kx1 views vector, defaults to None
        :type Q: np.ndarray or pd.DataFrame, optional
        :param P: KxN picking matrix, defaults to None
        :type P: np.ndarray or pd.DataFrame, optional
        :param omega: KxK view uncertainty matrix (diagonal), defaults to None
        :type omega: np.ndarray or Pd.DataFrame, optional
        :param tau: the weight-on-views scalar (default is 0.05)
        :type tau: float, optional
        """
        # No typechecks because we assume this is the output of a pypfopt method.
        self.cov_matrix = cov_matrix.values
        # Initialise BaseOptimizer
        super().__init__(len(cov_matrix), cov_matrix.columns)

        # Views
        if absolute_views is not None:
            self.Q, self.P = self._parse_views(absolute_views)
        else:
            if isinstance(Q, (pd.Series, pd.DataFrame)):
                self.Q = Q.values.reshape(-1, 1)
            elif isinstance(Q, np.ndarray):
                self.Q = Q.reshape(-1, 1)
            else:
                raise TypeError("Q must be an array or dataframe")

            if isinstance(P, pd.DataFrame):
                self.P = P.values
            elif isinstance(P, np.ndarray):
                self.P = P
            elif len(self.Q) == self.n_assets:
                # If a view on every asset is provided, P defaults
                # to the identity matrix.
                self.P = np.eye(self.n_assets)
            else:
                raise TypeError("P must be an array or dataframe")

        if pi is None:
            warnings.warn("Running Black-Litterman with no prior.")
            self.pi = np.zeros((self.n_assets, 1))
        else:
            if isinstance(pi, (pd.Series, pd.DataFrame)):
                self.pi = pi.values.reshape(-1, 1)
            elif isinstance(pi, np.ndarray):
                self.pi = pi.reshape(-1, 1)
            else:
                raise TypeError("pi must be an array or series")

        if tau <= 0 or tau > 1:
            raise ValueError("tau should be between 0 and 1")
        self.tau = tau

        if omega is None:
            self.omega = BlackLittermanModel._default_omega(
                self.cov_matrix, self.P, self.tau
            )
        else:
            if isinstance(omega, pd.DataFrame):
                self.omega = omega.values
            elif isinstance(omega, np.ndarray):
                self.omega = omega
            else:
                raise TypeError("self.omega must be a square array or dataframe")

        # Make sure all dimensions work
        self._check_attribute_dimensions()

    def _parse_views(self, absolute_views):
        """
        Given a collection (dict or series) of absolute views, construct
        the appropriate views vector and picking matrix. The views must
        be a subset of the tickers in the covariance matrix.

        {"AAPL": 0.20, "GOOG": 0.12, "XOM": -0.30}

        :param absolute_views: absolute views on asset performances
        :type absolute_views: dict, pd.Series
        """
        if not isinstance(absolute_views, (dict, pd.Series)):
            raise TypeError("views should be a dict or pd.Series")
        # Coerce to series
        views = pd.Series(absolute_views)
        # Q is easy to construct
        Q = views.values.reshape(-1, 1)
        # P maps views to the universe.
        P = np.zeros((len(Q), self.n_assets))
        for i, view_ticker in enumerate(views.keys()):
            try:
                P[i, list(self.tickers).index(view_ticker)] = 1
            except ValueError:
                #  Could make this smarter by just skipping
                raise ValueError("Providing a view on an asset not in the universe")
        return Q, P

    @staticmethod
    def _default_omega(cov_matrix, P, tau):
        """
        If the uncertainty matrix omega is not provided, we calculate using the method of
        He and Litterman (1999), such that the ratio omega/tau is proportional to the
        variance of the view portfolio.

        :return: KxK diagonal uncertainty matrix
        :rtype: np.ndarray
        """
        return np.diag(np.diag(tau * P @ cov_matrix @ P.T))

    def _check_attribute_dimensions(self):
        """
        Helper method to ensure that all of the attributes created by the initialiser
        have the correct dimensions, to avoid linear algebra errors later on.

        :raises ValueError: if there are incorrect dimensions.
        """
        try:
            N = self.n_assets
            K = len(self.Q)
            assert self.pi.shape == (N, 1)
            assert self.P.shape == (K, N)
            assert self.omega.shape == (K, K)
            assert self.cov_matrix.shape == (N, N)  # redundant
        except AssertionError:
            raise ValueError("Some of the inputs have incorrect dimensions.")

    def bl_returns(self):
        """
        Calculate the posterior estimate of the returns vector,
        given views on some assets.

        :return: posterior returns vector
        :rtype: pd.Series
        """
        omega_inv = np.diag(1 / np.diag(self.omega))
        P_omega_inv = self.P.T @ omega_inv
        tau_sigma_inv = np.linalg.inv(self.tau * self.cov_matrix)

        # Solve the linear system rather thatn invert everything
        A = tau_sigma_inv + P_omega_inv @ self.P
        b = tau_sigma_inv.dot(self.pi) + P_omega_inv.dot(self.Q)
        x = np.linalg.solve(A, b)
        return pd.Series(x.flatten(), index=self.tickers)

    def bl_cov(self):
        """
        Calculate the posterior estimate of the covariance matrix,
        given views on some assets. Based on He and Litterman (2002).

        :return: posterior covariance matrix
        :rtype: pd.DataFrame
        """
        omega_inv = np.diag(1.0 / np.diag(self.omega))
        P_omega_inv = self.P.T @ omega_inv
        tau_cov_matrix_inv = np.linalg.inv(self.tau * self.cov_matrix)
        M = np.linalg.inv(tau_cov_matrix_inv + P_omega_inv @ self.P)
        posterior_cov = self.cov_matrix + M
        return pd.DataFrame(posterior_cov, index=self.tickers, columns=self.tickers)

    def bl_weights(self):
        pass
