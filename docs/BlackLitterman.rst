.. _black-litterman:

################
Black-Litterman
################


Though the algorithm is relatively simple, BL proved to be a challenge from a software
engineering perspective because it's not quite clear how best to fit it into PyPortfolioOpt's
API. The full discussion can be found on a `Github issue thread <https://github.com/robertmartin8/PyPortfolioOpt/issues/48>`_

I ultimately decided that though BL is not technically an optimiser, it didn't make sense to
split up its methods into `expected_returns` or `risk_models`. I have thus made it an independent
module and owing to the comparatively extensive theory, have given it a dedicated documentation page.


The dimensionality of the priors must be the same as the dimensionality of the universe!


explain different letters


default tau = 0.05 as given by Black Litterman in Detail (walters)

further discussion here:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1701467
