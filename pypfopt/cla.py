"""
The ``cla`` module houses the CLA class, which
generates optimal portfolios using the Critical Line Algorithm as implemented
by Marcos Lopez de Prado.
"""

import numbers
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as sco
from . import objective_functions, base_optimizer


def _infnone(x):
    return float("-inf") if x is None else x


class CLA(base_optimizer.BaseOptimizer):
    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1)):
        """
        :param expected_returns: expected returns for each asset. Set to None if
                                 optimising for volatility only.
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1).
                              Must be changed to (-1, 1) for portfolios with shorting.
        :type weight_bounds: tuple (float, float) or (list/ndarray, list/ndarray)
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        :raises TypeError: if ``cov_matrix`` is not a dataframe or array
        """

        # Initialize the class
        self.mean = np.array(expected_returns).reshape((len(expected_returns), 1))
        if (self.mean == np.ones(self.mean.shape) * self.mean.mean()).all():
            self.mean[-1, 0] += 1e-5
        self.expected_returns = self.mean.reshape((len(self.mean),))
        self.cov_matrix = np.asarray(cov_matrix)
        if isinstance(weight_bounds[0], numbers.Real):
            self.lB = np.ones(self.mean.shape) * weight_bounds[0]
        else:
            self.lB = np.array(weight_bounds[0]).reshape(self.mean.shape)
        if isinstance(weight_bounds[0], numbers.Real):
            self.uB = np.ones(self.mean.shape) * weight_bounds[1]
        else:
            self.uB = np.array(weight_bounds[1]).reshape(self.mean.shape)
        self.w = []  # solution
        self.l = []  # lambdas
        self.g = []  # gammas
        self.f = []  # free weights

        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        else:
            tickers = list(range(len(self.mean)))
        super().__init__(len(tickers), tickers)

    def solve(self):
        # Compute the turning points,free sets and weights
        f, w = self.init_algo()
        self.w.append(np.copy(w))  # store solution
        self.l.append(None)
        self.g.append(None)
        self.f.append(f[:])
        while True:
            # 1) case a): Bound one free weight
            l_in = None
            if len(f) > 1:
                covarF, covarFB, meanF, wB = self.get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
                j = 0
                for i in f:
                    l, bi = self.compute_lambda(
                        covarF_inv, covarFB, meanF, wB, j, [self.lB[i], self.uB[i]]
                    )
                    if _infnone(l) > _infnone(l_in):
                        l_in, i_in, bi_in = l, i, bi
                    j += 1
            # 2) case b): Free one bounded weight
            l_out = None
            if len(f) < self.mean.shape[0]:
                b = self.get_b(f)
                for i in b:
                    covarF, covarFB, meanF, wB = self.get_matrices(f + [i])
                    covarF_inv = np.linalg.inv(covarF)
                    l, bi = self.compute_lambda(
                        covarF_inv,
                        covarFB,
                        meanF,
                        wB,
                        meanF.shape[0] - 1,
                        self.w[-1][i],
                    )
                    if (self.l[-1] == None or l < self.l[-1]) and l > _infnone(l_out):
                        l_out, i_out = l, i
            if (l_in == None or l_in < 0) and (l_out == None or l_out < 0):
                # 3) compute minimum variance solution
                self.l.append(0)
                covarF, covarFB, meanF, wB = self.get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
                meanF = np.zeros(meanF.shape)
            else:
                # 4) decide lambda
                if _infnone(l_in) > _infnone(l_out):
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f.append(i_out)
                covarF, covarFB, meanF, wB = self.get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
            # 5) compute solution vector
            wF, g = self.compute_w(covarF_inv, covarFB, meanF, wB)
            for i in range(len(f)):
                w[f[i]] = wF[i]
            self.w.append(np.copy(w))  # store solution
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1] == 0:
                break
        # 6) Purge turning points
        self.purge_num_err(10e-10)
        self.purge_excess()

    def init_algo(self):
        # Initialize the algo
        # 1) Form structured array
        a = np.zeros((self.mean.shape[0]), dtype=[("id", int), ("mu", float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]  # dump array into list
        # fill structured array
        a[:] = list(zip(list(range(self.mean.shape[0])), b))
        # 2) Sort structured array
        b = np.sort(a, order="mu")
        # 3) First free weight
        i, w = b.shape[0], np.copy(self.lB)
        while sum(w) < 1:
            i -= 1
            w[b[i][0]] = self.uB[b[i][0]]
        w[b[i][0]] += 1 - sum(w)
        return [b[i][0]], w

    def compute_bi(self, c, bi):
        if c > 0:
            bi = bi[1][0]
        if c < 0:
            bi = bi[0][0]
        return bi

    def compute_w(self, covarF_inv, covarFB, meanF, wB):
        # 1) compute gamma
        onesF = np.ones(meanF.shape)
        g1 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        g2 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        if wB is None:
            g, w1 = float(-self.l[-1] * g1 / g2 + 1 / g2), 0
        else:
            onesB = np.ones(wB.shape)
            g3 = np.dot(onesB.T, wB)
            g4 = np.dot(covarF_inv, covarFB)
            w1 = np.dot(g4, wB)
            g4 = np.dot(onesF.T, w1)
            g = float(-self.l[-1] * g1 / g2 + (1 - g3 + g4) / g2)
        # 2) compute weights
        w2 = np.dot(covarF_inv, onesF)
        w3 = np.dot(covarF_inv, meanF)
        return -w1 + g * w2 + self.l[-1] * w3, g

    def compute_lambda(self, covarF_inv, covarFB, meanF, wB, i, bi):
        # 1) C
        onesF = np.ones(meanF.shape)
        c1 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        c2 = np.dot(covarF_inv, meanF)
        c3 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        c4 = np.dot(covarF_inv, onesF)
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            return None, None
        # 2) bi
        if type(bi) == list:
            bi = self.compute_bi(c, bi)
        # 3) Lambda
        if wB is None:
            # All free assets
            return float((c4[i] - c1 * bi) / c), bi
        else:
            onesB = np.ones(wB.shape)
            l1 = np.dot(onesB.T, wB)
            l2 = np.dot(covarF_inv, covarFB)
            l3 = np.dot(l2, wB)
            l2 = np.dot(onesF.T, l3)
            return float(((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c), bi

    def get_matrices(self, f):
        # Slice covarF,covarFB,covarB,meanF,meanB,wF,wB
        covarF = self.reduce_matrix(self.cov_matrix, f, f)
        meanF = self.reduce_matrix(self.mean, f, [0])
        b = self.get_b(f)
        covarFB = self.reduce_matrix(self.cov_matrix, f, b)
        wB = self.reduce_matrix(self.w[-1], b, [0])
        return covarF, covarFB, meanF, wB

    def get_b(self, f):
        return self.diff_lists(list(range(self.mean.shape[0])), f)

    def diff_lists(self, list1, list2):
        return list(set(list1) - set(list2))

    def reduce_matrix(self, matrix, listX, listY):
        # Reduce a matrix to the provided list of rows and columns
        if len(listX) == 0 or len(listY) == 0:
            return
        matrix_ = matrix[:, listY[0] : listY[0] + 1]
        for i in listY[1:]:
            a = matrix[:, i : i + 1]
            matrix_ = np.append(matrix_, a, 1)
        matrix__ = matrix_[listX[0] : listX[0] + 1, :]
        for i in listX[1:]:
            a = matrix_[i : i + 1, :]
            matrix__ = np.append(matrix__, a, 0)
        return matrix__

    def purge_num_err(self, tol):
        # Purge violations of inequality constraints (associated with ill-conditioned covar matrix)
        i = 0
        while True:
            flag = False
            if i == len(self.w):
                break
            if abs(sum(self.w[i]) - 1) > tol:
                flag = True
            else:
                for j in range(self.w[i].shape[0]):
                    if (
                        self.w[i][j] - self.lB[j] < -tol
                        or self.w[i][j] - self.uB[j] > tol
                    ):
                        flag = True
                        break
            if flag == True:
                del self.w[i]
                del self.l[i]
                del self.g[i]
                del self.f[i]
            else:
                i += 1

    def purge_excess(self):
        # Remove violations of the convex hull
        i, repeat = 0, False
        while True:
            if repeat == False:
                i += 1
            if i == len(self.w) - 1:
                break
            w = self.w[i]
            mu = np.dot(w.T, self.mean)[0, 0]
            j, repeat = i + 1, False
            while True:
                if j == len(self.w):
                    break
                w = self.w[j]
                mu_ = np.dot(w.T, self.mean)[0, 0]
                if mu < mu_:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    repeat = True
                    break
                else:
                    j += 1

    def min_volatility(self):
        """Get the minimum variance solution"""
        if not self.w:
            self.solve()
        var = []
        for w in self.w:
            a = np.dot(np.dot(w.T, self.cov_matrix), w)
            var.append(a)
        # return min(var)**.5, self.w[var.index(min(var))]
        self.weights = self.w[var.index(min(var))].reshape((self.n_assets,))
        return dict(zip(self.tickers, self.weights))

    def max_sharpe(self):
        """Get the max Sharpe ratio portfolio"""
        if not self.w:
            self.solve()
        # 1) Compute the local max SR portfolio between any two neighbor turning points
        w_sr, sr = [], []
        for i in range(len(self.w) - 1):
            w0 = np.copy(self.w[i])
            w1 = np.copy(self.w[i + 1])
            kargs = {"minimum": False, "args": (w0, w1)}
            a, b = self.golden_section(self.eval_sr, 0, 1, **kargs)
            w_sr.append(a * w0 + (1 - a) * w1)
            sr.append(b)
        # return max(sr), w_sr[sr.index(max(sr))]
        self.weights = w_sr[sr.index(max(sr))].reshape((self.n_assets,))
        return dict(zip(self.tickers, self.weights))

    def eval_sr(self, a, w0, w1):
        # Evaluate SR of the portfolio within the convex combination
        w = a * w0 + (1 - a) * w1
        b = np.dot(w.T, self.mean)[0, 0]
        c = np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5
        return b / c

    def golden_section(self, obj, a, b, **kargs):
        # Golden section method. Maximum if kargs['minimum']==False is passed
        from math import log, ceil

        tol, sign, args = 1.0e-9, 1, None
        if "minimum" in kargs and kargs["minimum"] == False:
            sign = -1
        if "args" in kargs:
            args = kargs["args"]
        numIter = int(ceil(-2.078087 * log(tol / abs(b - a))))
        r = 0.618033989
        c = 1.0 - r
        # Initialize
        x1 = r * a + c * b
        x2 = c * a + r * b
        f1 = sign * obj(x1, *args)
        f2 = sign * obj(x2, *args)
        # Loop
        for i in range(numIter):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        if f1 < f2:
            return x1, sign * f1
        else:
            return x2, sign * f2

    def efficient_frontier(self, points):
        """Get the efficient frontier"""
        mu, sigma, weights = [], [], []
        # remove the 1, to avoid duplications
        a = np.linspace(0, 1, points / len(self.w))[:-1]
        b = list(range(len(self.w) - 1))
        for i in b:
            w0, w1 = self.w[i], self.w[i + 1]
            if i == b[-1]:
                # include the 1 in the last iteration
                a = np.linspace(0, 1, points / len(self.w))
            for j in a:
                w = w1 * j + (1 - j) * w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T, self.mean)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5)
        return mu, sigma, weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        return base_optimizer.portfolio_performance(
            self.expected_returns,
            self.cov_matrix,
            self.weights,
            verbose,
            risk_free_rate,
        )
