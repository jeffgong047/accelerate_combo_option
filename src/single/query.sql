
    SELECT Date, strike, bid AS best_bid, ask AS best_offer, vol AS implied_volatility, delta, gamma, vega
    FROM option_chain
    WHERE act_symbol = 'A' AND expiration BETWEEN '2019-02-22' AND '2019-03-29'
    