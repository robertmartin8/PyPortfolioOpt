<p align="center">
    <img width=60% src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/logo_v1.png">
</p>

<!-- buttons -->
<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
    <a href="https://pypi.org/project/PyPortfolioOpt/">
        <img src="https://img.shields.io/badge/pypi-v0.4.3-brightgreen.svg"
            alt="pypi"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"
            alt="MIT license"></a> &nbsp;
    <a href="https://github.com/robertmartin8/PyPortfolioOpt/graphs/commit-activity">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg"
            alt="issues"></a> &nbsp;
    <a href="https://pyportfolioopt.readthedocs.io/en/latest/">
        <img src="https://img.shields.io/badge/docs-passing-brightgreen.svg"
            alt="docs"></a> &nbsp;
    <a href="https://travis-ci.org/robertmartin8/PyPortfolioOpt">
        <img src="https://travis-ci.org/robertmartin8/PyPortfolioOpt.svg?branch=master"
            alt="travis"></a> &nbsp;
</p>

<!-- content -->

PyPortfolioOpt is a library that implements widely-used classical portfolio optimisation techniques, with a number of experimental features. It is **extensive** yet easily **extensible**, and can be useful for both the casual investor and the serious practitioner.

Whether you are a fundamentals-oriented investor who has identified a handful of undervalued picks, or an algorithmic trader who has a basket of interesting signals, PyPortfolioOpt can help you combine your alpha-generators in a risk-efficient way.

Head over to the [documentation on ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/) to get an in-depth look at the project, or continue below to check out some examples.

<center>
<img src="https://github.com/robertmartin8/PyPortfolioOpt/blob/master/media/conceptual_flowchart_v1.png" style="width:70%;"/>
</center>

## Table of contents

- [Table of contents](#table-of-contents)
- [Getting started](#getting-started)
  - [For development](#for-development)
- [A quick example](#a-quick-example)
- [An overview of classical portfolio optimisation methods](#an-overview-of-classical-portfolio-optimisation-methods)
- [Features](#features)
  - [Expected returns](#expected-returns)
  - [Risk models (covariance)](#risk-models-covariance)
  - [Objective functions](#objective-functions)
  - [Optional parameters](#optional-parameters)
  - [Other optimisers](#other-optimisers)
- [Advantages over existing implementations](#advantages-over-existing-implementations)
- [Project principles and design decisions](#project-principles-and-design-decisions)
- [Roadmap](#roadmap)
- [Testing](#testing)
- [Contributing](#contributing)
- [Getting in touch](#getting-in-touch)

## Getting started

This project is available on PyPI, meaning that you can just:

```bash
pip install PyPortfolioOpt
```

However, I have since been converted to `poetry`, so my current recommendation is to get yourself set up with [poetry](https://github.com/sdispater/poetry) then just run

```bash
poetry add PyPortfolioOpt
```

Otherwise, clone/download the project and in the project directory run:

```bash
python setup.py install
```

### For development

If you would like to make major changes to integrate this with your proprietary system, it probably makes sense to clone this repository and to just use the source code.

```bash
git clone https://github.com/robertmartin8/PyPortfolioOpt
```

Alternatively, you could try:

```bash
pip install -e git+https://github.com/robertmartin8/PyPortfolioOpt.git
```

## A quick example

Here is an example on real life stock data, demonstrating how easy it is to find the long-only portfolio that maximises the Sharpe ratio (a measure of risk adjusted returns).

```python
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Read in price data
df = pd.read_csv("tests/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
```

This outputs the following weights:

```txt
{'GOOG': 0.01269,
 'AAPL': 0.09202,
 'FB': 0.19856,
 'BABA': 0.09642,
 'AMZN': 0.07158,
 'GE': 0.0,
 'AMD': 0.0,
 'WMT': 0.0,
 'BAC': 0.0,
 'GM': 0.0,
 'T': 0.0,
 'UAA': 0.0,
 'SHLD': 0.0,
 'XOM': 0.0,
 'RRC': 0.0,
 'BBY': 0.06129,
 'MA': 0.24562,
 'PFE': 0.18413,
 'JPM': 0.0,
 'SBUX': 0.03769}

Expected annual return: 33.0%
Annual volatility: 21.7%
Sharpe Ratio: 1.43
```

Instead of just stopping here, PyPortfolioOpt provides a method which allows you to convert the above continuous weights to an actual allocation that you could buy. Just enter the most recent prices, and the desired portfolio size ($10000 in this example):

```python
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


latest_prices = get_latest_prices(df)

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
```

```txt
11 out of 20 tickers were removed
Discrete allocation: {'GOOG': 0, 'AAPL': 5, 'FB': 11, 'BABA': 5, 'AMZN': 1,
                      'BBY': 7, 'MA': 14, 'PFE': 50, 'SBUX': 5}
Funds remaining: $8.42
```

*Disclaimer: nothing about this project constitues investment advice, and the author bears no responsibiltiy for your subsequent investment decisions. Please refer to the [license](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/LICENSE.txt) for more information.*

## An overview of classical portfolio optimisation methods

Harry Markowitz's 1952 paper is the undeniable classic, which turned portfolio optimisation from an art into a science. The key insight is that by combining assets with different expected returns and volatilities, one can decide on a mathematically optimal allocation which minimises the risk for a target return – the set of all such optimal portfolios is referred to as the **efficient frontier**.

Although much development has been made in the subject, more than half a century later, Markowitz's core ideas are still fundamentally important, and see daily use in many portfolio management firms. The main drawback of mean-variance optimisation is that the theoretical treatment requires knowledge of the expected returns and the future risk-characteristics (covariance) of the assets. Obviously, if we knew the expected returns of a stock life would be much easier, but the whole game is that stock returns are notoriously hard to forecast. As a substitute, we can derive estimates of the expected return and covariance based on historical data – though we do lose the theoretical guarantees provided by Markowitz, the closer our estimates are to the real values, the better our portfolio will be.

Thus this project provides four major sets of functionality (though of course they are intimately related)

- Estimate of expected returns
- Estimate of the covariance of assets
- Objective functions to be optimised
- Parameters for the efficient frontier

## Features

In this section, we detail PyPortfolioOpt's current available functionality as per the above breakdown. More examples are offered in `examples.py`.

A far more comprehensive version of this can be found on [ReadTheDocs](https://pyportfolioopt.readthedocs.io/en/latest/), as well as possible extensions for more advanced users.

### Expected returns

- Mean historical returns:
    - the simplest and most common approach, which states that the expected return of each asset is equal to the mean of its historical returns.
    - easily interpretable and very intuitive
- Exponentially weighted mean historical returns:
    - similar to mean historical returns, except it gives exponentially more weight to recent prices
    - it is likely the case that an asset's most recent returns hold more weight than returns from 10 years ago when it comes to estimating future returns.

### Risk models (covariance)

The covariance matrix encodes not just the volatility of an asset, but also how it correlated to other assets. This is important because in order to reap the benefits of diversification (and thus increase return per unit risk), the assets in the portfolio should be as uncorrelated as possible.

- Sample covariance matrix:
    - an unbiased estimate of the covariance matrix
    - relatively easy to compute
    - the de facto standard for many years
    - however, it has a high estimation error, which is particularly dangerous in mean-variance optimisation because the optimiser is likely to give excess weight to these erroneous estimates.
- Semicovariance: a measure of risk that focuses on downside variation.
- Exponential covariance: an improvement over sample covariance that gives more weight to recent data
- Covariance shrinkage: techniques that involve combining the sample covariance matrix with a structured estimator, in order to reduce the effect of erroneous weights. PyPortfolioOpt provides wrappers around the efficient vectorised implementations provided by `sklearn.covariance`.
    - manual shrinkage
    - Ledoit Wolf shrinkage, which chooses an optimal shrinkage parameter. We offer three shrinkage targets: `constant_variance`, `single_factor`, and `constant_correlation`.
    - Oracle Approximating Shrinkage
- Minimum Covariance Determinant:
    - a robust estimate of the covariance
    - implemented in `sklearn.covariance`

### Objective functions

- Maximum Sharpe ratio: this results in a *tangency portfolio* because on a graph of returns vs risk, this portfolio corresponds to the tangent of the efficient frontier that has a y-intercept equal to the risk-free rate. This is the default option because it finds the optimal return per unit risk.
- Minimum volatility. This may be useful if you're trying to get an idea of how low the volatility *could* be, but in practice it makes a lot more sense to me to use the portfolio that maximises the Sharpe ratio.
- Efficient return, a.k.a. the Markowitz portfolio, which minimises risk for a given target return – this was the main focus of Markowitz 1952
- Efficient risk: the Sharpe-maximising portfolio for a given target risk.
- Condiitional value-at-risk: a measure of tail loss

### Optional parameters

- Long/short: by default all of the mean-variance optimisation methods in PyPortfolioOpt are long-only, but they can be initialised to allow for short positions by changing the weight bounds:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
```

- Market neutrality: for the `efficient_risk` and `efficient_return` methods, PyPortfolioOpt provides an option to form a market neutral portfolio (i.e weights sum to zero). This is not possible for the max Sharpe portfolio and the min volatility portfolio because in those cases because they are not invariant with respect to leverage. Market neutrality requires negative weights:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
ef.efficient_return(target_return=0.2, market_neutral=True)
```

- Minimum/maximum position size: it may be the case that you want no security to form more than 10% of your portfolio. This is easy to encode:

```python
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
```

- L2 Regularisation: this is a novel experimental feature which can be used to reduce the number of negligible weights for any of the objective functions. Essentially, it adds a penalty (parameterised by `gamma`) on small weights, with a term that looks just like L2 regularisation in machine learning. It may be necessary to trial a number of `gamma` values to achieve the desired number of non-negligible weights. For the test portfolio of 20 securities, `gamma ~ 1` is sufficient

```python
ef = EfficientFrontier(mu, S, gamma=1)
ef.max_sharpe()
```

### Other optimisers

The features above mostly pertain to efficient frontier optimisation via quadratic programming. However, we offer different optimisers as well:

- Hierarchical Risk Parity, using clustering algorithms to choose uncorrelated assets
- Markowitz's critical line algorithm (CLA)

Please refer to the [documentation](https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimisers.html) for more.

## Advantages over existing implementations

- Includes both classical methods (Markowitz 1952), suggested best practices
  (e.g covariance shrinkage), along with many recent developments and novel
  features, like L2 regularisation, shrunk covariance, hierarchical risk parity.
- Native support for pandas dataframes: easily input your daily prices data.
- Extensive practical tests, which use real-life data.
- Easy to combine with your own proprietary strategies and models.
- Robust to missing data, and price-series of different lengths (e.g FB data
  only goes back to 2012 whereas AAPL data goes back to 1980).

## Project principles and design decisions

- It should be easy to swap out individual components of the optimisation process
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimisation unless it can be practically
  applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this,
  I have deferred **all** formatting decisions to [Black](https://github.com/ambv/black).

## Roadmap

Feel free to raise an issue requesting any new features – here are some of the things I want to implement:

- Custom utility functions, including risk aversion
- Plotting the efficient frontier.
- More optimisation goals, including the Calmar Ratio, Sortino Ratio, etc.
- Monte Carlo optimisation with custom distributions
- Black-Litterman portfolio selection
- Improved CVaR optimisation using linear programming.

## Testing

Tests are written in pytest (much more intuitive than `unittest` and the variants in my opinion), and I have tried to ensure close to 100% coverage. Run the tests by navigating to the package directory and simply running `pytest` on the command line.

PyPortfolioOpt provides a test dataset of daily returns for 20 tickers:

```python
['GOOG', 'AAPL', 'FB', 'BABA', 'AMZN', 'GE', 'AMD', 'WMT', 'BAC', 'GM',
'T', 'UAA', 'SHLD', 'XOM', 'RRC', 'BBY', 'MA', 'PFE', 'JPM', 'SBUX']
```

 These tickers have been informally selected to meet a number of criteria:

- reasonably liquid
- different performances and volatilities
- different amounts of data to test robustness

Currently, the tests have not explored all of the edge cases and combinations
of objective functions and parameters. However, each method and parameter has
been tested to work as intended.

## Contributing

Contributions are *most welcome*. Have a look at the [Contribution Guide](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/CONTRIBUTING.md) for more.

## Getting in touch

If you would like to reach out for any reason, be it consulting opportunities or just for a chat, please do so via the [form](https://reasonabledeviations.com/about/) on my website.
