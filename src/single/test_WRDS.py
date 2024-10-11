import wrds
import pandas as pd
# Connect to WRDS (enter your username/password when prompted)
db = wrds.Connection()

# Example: Query daily option prices for a specific stock (e.g., Apple, secid='10463')
query = """
    SELECT date, secid, strike_price, best_bid, best_offer, impl_volatility, open_interest, delta, gamma, vega
    FROM optionm.opprcd
    WHERE secid = '10463' -- AAPL
    AND date >= '2019-01-01' AND date <= '2019-12-31'
"""
option_data = db.raw_sql(query)

# Display the first few rows
print(option_data.head())


import numpy as np

# Ensure the 'date' column is in datetime format
option_data['date'] = pd.to_datetime(option_data['date'])

# Pivot the data to get time-series format (group by date)
# Example: Getting the average implied volatility and delta per day
time_series_data = option_data.groupby('date').agg({
    'best_bid': 'mean',
    'best_offer': 'mean',
    'impl_volatility': 'mean',
    'delta': 'mean',
    'gamma': 'mean',
    'vega': 'mean',
    'open_interest': 'sum'
}).reset_index()

# Check the transformed time-series data
print(time_series_data.head())
