def test_import_modules():
    from pypfopt import (
        base_optimizer,
        black_litterman,
        cla,
        discrete_allocation,
        exceptions,
        expected_returns,
        hierarchical_portfolio,
        objective_functions,
        plotting,
        risk_models,
    )


def test_explicit_import():
    from pypfopt.black_litterman import (
        market_implied_prior_returns,
        market_implied_risk_aversion,
        BlackLittermanModel,
    )
    from pypfopt.cla import CLA
    from pypfopt.discrete_allocation import get_latest_prices, DiscreteAllocation
    from pypfopt.efficient_frontier import (
        EfficientFrontier,
        EfficientSemivariance,
        EfficientCVaR,
    )
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage


def test_import_toplevel():
    from pypfopt import (
        market_implied_prior_returns,
        market_implied_risk_aversion,
        BlackLittermanModel,
        CLA,
        get_latest_prices,
        DiscreteAllocation,
        EfficientFrontier,
        EfficientSemivariance,
        EfficientCVaR,
        HRPOpt,
        CovarianceShrinkage,
    )
