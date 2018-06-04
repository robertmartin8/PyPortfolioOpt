<p align="center">
    <img width=60% src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/logo_v1.png">
</p>

<!-- issues, build status -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg?style=flat-square"
            alt="python"></a> &nbsp
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp
    <a href="https://github.com/robertmartin8/PyPortfolioOpt/graphs/commit-activity">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp
</p>

## Introduction

**This project is still in development, please check back in one week!**

PyPortfolioOpt is a library that contains widely-used classical portfolio optimisation techniques, with
a number of novel features. It is extensive, yet easily extensible – the methods provided herein are useful for both the casual investor and the serious practitioner.

Whether you are a fundamentals-oriented investor who has identified a handful of undervalued picks, or an algorithmic trader who has a basket of signals, portfolio optimisation methods are important to help you combine your alpha-generators in a risk-efficient way. `PyPortfolioOpt` is designed to be easily compatible with your own strategy: just feed in the expected risk and return of the assets or signals, and let PyPortfolioOpt weight them the *right* way.

<center>
<img src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/conceptual_flowchart_v1.png" style="width:90%;"/>
</center>

Here is an example on real life stock data, demonstrating how easy it is to find the long-only portfolio that maximises the Sharpe Ratio (a measure of risk adjusted returns).

```python
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in test data
df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Calculate the efficient weights
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()  # optimise for the Sharpe Ratio
print(weights)
ef.portfolio_performance(verbose=True)
```

This outputs the following weights:

```txt
{'GOOG': 0.012686476493307005,
 'AAPL': 0.09201504086768657,
 'FB': 0.19856142332208873,
 'BABA': 0.0964204513428177,
 'AMZN': 0.07158459468700305,
 'GE': 9.865059336174564e-17,
 'AMD': 2.9475583731962473e-16,
 'WMT': 6.367602325244733e-17,
 'BAC': 1.2450912524798556e-16,
 'GM': 3.6652897677048814e-17,
 'T': 6.7707051460844506e-18,
 'UAA': 1.246670524382067e-17,
 'SHLD': 2.021027903443432e-16,
 'XOM': 1.976362521670158e-17,
 'RRC': 2.0602865427398947e-17,
 'BBY': 0.061286677246206595,
 'MA': 0.24562306492773736,
 'PFE': 0.18412827470141765,
 'JPM': 1.1654588908289105e-17,
 'SBUX': 0.03769399641173482}

Expected annual return: 33.0%
Annual volatility: 21.7
Sharpe Ratio: 1.43
```

Instead of just stopping here, `PyPortfolioOpt` provides a method which allows you to convert the above continuous weights to an actual allocation, which you can trade on. Just enter the most recent prices, and the desired portfolio size ($10000 in this example):

```python
from pypfopt import discrete_allocation

latest_prices = discrete_allocation.get_latest_prices(df)
allocation, leftover = discrete_allocation.portfolio(weights, latest_prices, 10000)
print(allocation)
print("Leftover: ${:.2f}".format(leftover))
```

```txt
{'MA': 14, 'FB': 12, 'PFE': 51, 'BABA': 5, 'AAPL': 5, 'AMZN': 0, 'BBY': 9, 'SBUX': 6, 'GOOG': 1}
```

### Getting started

Within this week, the project will become available on PyPI, so it will just be a matter of:

```bash
pip install PyPortfolioOpt
```

Until then, it is probably easiest to clone/download the project, and place the `pypfopt` folder into your working directory.

## Principles guiding design decisions

- It should be easy to swap out individual components of the optimisation process with the user's proprietary improvements.
- User-friendliness is **everything**.
- There is no point in portfolio optimisation unless it can be practically applied to real asset prices.
- Everything that has been implemented should be tested
- Inline documentation is good: dedicated documentation hosted on readthedocs is better.
- Formatting should never get in the way of good code: because of this I have deferred **all** formatting decisions to [Black](https://github.com/ambv/black). Initially some of its decisions irritated me, but it is extremely consistent and actually quite elegant.

### Advantages over existing implementations

- Includes both classical methods (Markowitz 1952), and more recent developments (covariance shrinkage), as well as experimental features such as L2-regularised weights.
- Native support for pandas dataframes: easily input your daily prices data.
- Clear and comprehensive documentation, hosted on readthedocs (coming soon)
- Extensive practical tests, which use real-life data.
- Easy to combine with your own proprietary strategies and models.
- Robust to missing data, and price-series of different lengths (e.g FB data only goes back to 2012, whereas AAPL data goes back to 1980).

*Disclaimer: nothing about this project constitues investment advice, and the author bears no responsibiltiy for your subsequent investment decisions. Please refer to the [license](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/LICENSE.txt) for more information.*

## An overview of classical portfolio optimisation methods

Harry Markowitz's 1952 paper is the undeniable classic, which turned portfolio optimisation from an art into a science. The key insight is that by combining assets with different expected returns and volatilities, one can decide on a matehmatically optimal allocation, which can efficiently minimise the risk for a target return – the set of all such optimal portfolios is referred to as the **efficient frontier**.

Although much development has been made in the subject, more than half a century later, Markowitz's core ideas are still fundamentally important, and see daily use in many portfolio management firms. The main drawback of mean variance optimisation is that the theoretical treatment requires knowledge of the expected returns and the future risk-characteristics (covariance) of the assets. Obviously, if we knew the expected returns of a stock life would be much easier, but the whole game is that stock returns are notoriously hard to forecast. As a substitute, we can derive estimates of the expected return and covariance based on historical data – though we do lose the theoretical guarantees provided by Markowitz, the closer our estimates are to the real values, the better our portfolio will be.

Thus this project provides four major sets of functionality (though of course they are intimately related)

- Estimate of expected returns
- Estimate of the covariance of assets
- Objective functions to be optimised
- Parameters for the efficient frontier

## Features

In this section, we detail PyPortfolioOpt's current available functionality as per the above breakdown. Full examples are offered in `examples.py`.

### Expected returns

- Mean historical returns:
    - the simplest and most common approach, which states that the expected return of each asset is equal to the mean of its historical returns.
    - easily interpretable and very intuitive
- Exponentially weighted mean historical returns:
    - similar to mean historical returns, except it gives exponentially more weight to recent prices
    - it is likely the case that an asset's most recent returns hold more weight than returns from 10 years ago when it comes to estimating future returns.

### Covariance

The covariance matrix encodes not just the volatility of an asset, but also how it correlated to other assets. This is important, because in order to reap the benefits of diversification (and thus increase return per unit risk), the assets in the portfolio should be as uncorrelated as possible.

- Sample covariance matrix:
    - an unbiased estimate of the covariance matrix
    - relatively easy to compute
    - the de facto standard for many years
    - however, it has a high estimation error, which is particularly dangerous in mean-variance optimisation because the optimiser is likely to give excess weight to these erroneous estimates.
- Covariance shrinkage: techniques that involve combining the sample covariance matrix with a structured estimator, in order to reduce the effect of erroneous weights. `PyPortfolioOpt` provides wrappers around the efficient vectorised implementations provided by `sklearn.covariance`.
    - manual shrinkage
    - Ledoit Wolf shrinkage, which chooses an optimal shrinkage parameter
    - Oracle Approximating Shrinkage
- Minimum Covariance Determinant:
    - a robust estimate of the covariance
    - implemented in `sklearn.covariance`

### Objective functions

- Maximum Sharpe ratio: this is also called the *tangency portfolio*, because on a graph of returns vs risk, this portfolio corresponds to the tangent of the efficient frontier that has a y-intercept equal to the risk free rate. This is the default option, because it finds the optimal return per unit risk.
- Minimum volatility. This may be useful if you're trying to get an idea of how low the volatiltiy *could* be, but in practice it makes a lot more sense to me to use the portfolio that maximises the Sharpe ratio.
- Efficient return, a.k.a. the Markowitz portfolio, which minimises risk for a given target return – this was the main focus of Markowitz 1952
- Efficient risk: the Sharpe-maximising portfolio for a given target risk.

### Optional parameters

- Long/short: by default all of the mean variance optimisation methods in PyPortfolioOpt are long-only, but they can be initialised to allow for short positions by changing the weight bounds:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
```

- Market neutrality: for the `efficient_risk` and `efficient_return` methods, PyPortfolioOpt provides an option to form a market neutral portfolio (i.e weights sum to zero). This is not possible for the max Sharpe portfolio and the min volatility portfolio because in those cases because they are not invariant with respect to leverage. Market neturality requires negative weights:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
ef.efficient_return(target_return=0.2, market_neutral=True)
```

- Minimum/maximum position size: it may be the case that you want no security to form more than 10% of your portfolio. This is easy to encode:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
```

- L2 Regularisation: this is a novel experimental feature which can be used to reduce the number of negligible weights for any of the objective functions. Essentially, it adds a penalty (parameterised by `alpha`) on small weights, with a term that looks just like L2 regularisation in machine learning (except with the opposite sign). It may be necessary to trial a number of `alpha` values to achieve the desired number of non-neglibile weights. For the test portfolio of 20 securities, `alpha ~ 1` is sufficient

```python
ef = EfficientFrontier(mu, S)
ef.max_sharpe(alpha=1)
```

## Roadmap

- Custom utility functions, including risk aversion
- More optimisation goals, including the Calmar Ratio, Sortino Ratio, etc.
- Monte Carlo optimisation with custom distributions
- Black-Litterman portfolio selection
- Open-source backtests using either [Backtrader](https://www.backtrader.com/) or [Zipline](https://github.com/quantopian/zipline).
- Genetic optimisation methods
- Further support for different risk/return models, including constant correlation shrinkage.

## Testing

Tests are written in pytest (much more intuitive than `unittest` and the variants IMO), and I have tried to ensure close to 100% coverage. `PyPortfolioOpt` provides a test dataset of daily returns for 20 tickers:

```python
['GOOG', 'AAPL', 'FB', 'BABA', 'AMZN', 'GE', 'AMD', 'WMT', 'BAC', 'GM',
'T', 'UAA', 'SHLD', 'XOM', 'RRC', 'BBY', 'MA', 'PFE', 'JPM', 'SBUX']
```

 These tickers have been informally selected to meet a number of criteria:

- reasonably liquid
- different performances and volatilities
- different amounts of data to test robustness

Currently, the tests have not explored all of the edge cases, however I have investigated the experimental features like L2 regularisation. Additionally, the tests currently have not satisfactorily tested all combinations of of objective function and options.
