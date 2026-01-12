"""
Docstring for notebooks.01_market_data.003_worsedaysfwdreturn
This module contains code that looks at the forward return of the worst performing days in the market.


Thought process:
1. import necessary libraries
2. load market data. Assume SMP 500 as the best proxy for market performance
3. Calculate daily return for the last 5 years
4. Identify the worst 1% of daily returns
5. Calculate the forward return for these worst days over different time horizons (1 day, 5 days, 10 days, 20 days)
6. Analyze and visualize the results to see how the market performs after these worst days.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["^GSPC"]  # S&P 500 Index
mdata = yf.download(
    "^GSPC",
    start = "2020-01-01",
    end = "2024-06-01",
    auto_adjust= False,
    progress= False
)


px = mdata["Adj Close"].squeeze().dropna()
returns = px.pct_change().dropna()
print(type(returns))
q = 0.02
thr = returns.quantile(q)
worst_days_mask = returns <= thr

worst_days = returns[worst_days_mask].sort_values()

print(f"2% threshold: {thr:.4%} | worst days count: {worst_days.shape[0]}")

horizons = [1, 5, 21, 63]

output = pd.DataFrame({"returns_0": returns.loc[worst_days.index]})
                       
output["thresholds"] = thr

for h in horizons:
    output[f"fwd_{h}d"] = px.shift(-h).loc[worst_days.index] / px.loc[worst_days.index] - 1


output = output.dropna()
output["rank"] = output["returns_0"].rank(method="first")
output = output.sort_values("rank")

print(output)