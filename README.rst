##############
PyPortfolioOpt
##############

************
Introduction
************

PyPortfolioOpt is a simple library that contains widely used portfolio optimisation techniques, with 
a number of novel/experimental features.

*********************
Currently Implemented
*********************

Implemented

Efficient frontier

*******
Testing
*******

Test use a returns dataset using daily returns for 20 tickers. These tickers have been informally selected 
to meet a number of criteria

- reasonably liquid
- different performances and volatilities
- different amounts of data to test robustness


****************
Design decisions
****************

- Should be easy to swap out components to test
- Some robustness to missing data
