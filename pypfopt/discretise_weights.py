import numpy as np

# TODO discrete portfolio allocation input and return types


def discrete_portfolio_allocation(
    weights, min_allocation, portfolio_size, latest_prices
):
    """
    Generates a discrete allocation based on continuous weights, using a greedy algorithm,
    then stores in an instance variable as a list of tuples
    :return: 1. a dict containing the ticker and the number of shares that should be purchased,
                2. leftover funds
    """
    # Drop any companies with negligible weights
    nonzero_cont_allocation = [i for i in weights if i[1] > min_allocation]
    remaining_tickers = [i[0] for i in nonzero_cont_allocation]
    print(
        f"{len(weights) - len(nonzero_cont_allocation)} out of {len(weights)} tickers were removed"
    )
    print(f"Remaining tickers: {remaining_tickers}\n")
    nonzero_cont_allocation.sort(key=lambda x: x[1])
    nonzero_cont_allocation = nonzero_cont_allocation[::-1]
    available_funds = portfolio_size
    shares_purchased = []
    share_prices = []

    for pair in nonzero_cont_allocation:
        ticker, weight = pair
        share_price = latest_prices[ticker]

        n_shares = int(weight * portfolio_size / share_price)
        cost_basis = n_shares * share_price
        if cost_basis > available_funds:
            n_shares = int(available_funds // share_price)
            if n_shares == 0:
                print("Insufficient funds")
        available_funds -= cost_basis
        shares_purchased.append(n_shares)
        share_prices.append(share_price)

    # Second round
    while available_funds > 0:
        actual_weights = np.array(share_prices) * np.array(shares_purchased)
        actual_weights /= actual_weights.sum()
        ideal_weights = np.array([i[1] for i in nonzero_cont_allocation])
        deficit = ideal_weights - actual_weights

        idx = np.argmax(deficit)
        ticker, weight = nonzero_cont_allocation[idx]
        share_price = latest_prices[ticker]

        counter = 0
        while share_price > available_funds:
            # Find the second highest deficit and carry on
            deficit[idx] = 0
            idx = np.argmax(deficit)
            if deficit[idx] < 0 or counter == 10:
                break

            ticker, weight = nonzero_cont_allocation[idx]
            share_price = latest_prices[ticker]
            counter += 1

        if deficit[idx] < 0 or counter == 10:
            break
        shares_purchased[idx] += 1
        available_funds -= share_price

    print(f"Funds remaining: {available_funds:.2f}")

    # The instance variable is a list of tuples, while the returned value is a dict.
    num_shares = list(zip([i[0] for i in nonzero_cont_allocation], shares_purchased))
    return dict(num_shares), available_funds
