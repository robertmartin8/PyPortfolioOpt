
<p align="center"><img width=12.5% src="https://github.com/robertmartin8/blob/master/media/logo_v0.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v2.7%20%2F%20v3.6-blue.svg)
[![Build Status](https://travis-ci.org/anfederico/Clairvoyant.svg?branch=master)](https://travis-ci.org/anfederico/Clairvoyant)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![GitHub Issues](https://img.shields.io/github/issues/anfederico/Clairvoyant.svg)](https://github.com/anfederico/Clairvoyant/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Introduction

PyPortfolioOpt is a simple library that contains widely used portfolio optimisation techniques, with 
a number of novel/experimental features.

## Currently Implemented

- Efficient frontier

## Testing

Tests use a returns dataset using daily returns for 20 tickers. These tickers have been informally selected to meet a number of criteria

- reasonably liquid
- different performances and volatilities
- different amounts of data to test robustness

## Design decisions

- Should be easy to swap out components to test
- Some robustness to missing data
