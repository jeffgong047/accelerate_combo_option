import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from skopt import gp_minimize
from skopt.space import Real
import os
from sqlalchemy import create_engine
from utils import get_stock_price
import matplotlib.pyplot as plt

# Black-Scholes formula for European call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Likelihood function: The probability of observing the option price given a stock price
def likelihood(stock_price, observed_price, K, T, r, sigma):
    predicted_price = black_scholes_call(stock_price, K, T, r, sigma)
    return norm.pdf(observed_price, predicted_price, 1.0)

# Bayesian estimation of stock price with iterative updates
def update_estimate_with_observations(observed_prices, K, T, r, sigma, mu_prior, sigma_prior):
    # Iterate over all observed prices
    for observed_price in observed_prices:
        # Define the posterior function for current observation
        def posterior(stock_price):
            prior = norm.pdf(stock_price, mu_prior, sigma_prior)
            lik = likelihood(stock_price, observed_price, K, T, r, sigma)
            return prior * lik
        
        # Perform numerical optimization to find the stock price estimate
        result = minimize(lambda s: -posterior(s), mu_prior, bounds=[(1, 500)])
        
        # Update prior based on the posterior mean and variance
        mu_prior = result.x[0]
        sigma_prior = sigma_prior / 2  # Shrinking uncertainty as we get more observations
    print(mu_prior, sigma_prior)
    # Return the final estimate and prior information
    return mu_prior, sigma_prior

# Bayesian Optimization using Gaussian Processes
def bayesian_optimize_stock_price(observed_price, K, T, r, sigma, mu_prior, sigma_prior):
    def posterior(stock_price):
        prior = norm.pdf(stock_price, mu_prior, sigma_prior)
        lik = likelihood(stock_price, observed_price, K, T, r, sigma)
        return prior * lik

    def objective(stock_price):
        return -posterior(stock_price[0])
    
    # Perform Bayesian optimization with a Gaussian Process
    result = gp_minimize(
        objective,
        dimensions=[Real(1, 500, name="stock_price")],
        n_calls=20,  # Number of iterations
        random_state=42
    )
    
    return result.x[0], result

# Query DoltHub for options data using a system call
def fetch_option_data(stock, expiration_date_between):
    # Define SQL query to get all options that expire within the given date range
    username = 'root'  # Default username for Dolt
    password = ''      # Default password is usually empty for local
    host = 'localhost' # Assuming it's running locally
    port = '3306'      # Default MySQL/Dolt port
    database = 'options'  # Your Dolt database name

    # Create an engine to connect to the Dolt database
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    query = f"""
    SELECT Date, strike, bid AS best_bid, ask AS best_offer, vol AS implied_volatility, delta, gamma, vega
    FROM option_chain
    WHERE act_symbol = '{stock}' AND expiration BETWEEN '{expiration_date_between[0]}' AND '{expiration_date_between[1]}'
    """
    option_data = pd.read_sql(query, con=engine)
    option_data.to_csv(f'{stock}_option.csv', index=False)
    
    return option_data

# Transform options data into time-series format
def transform_option_data(option_data):
    # Convert 'date' column to datetime if not already
    option_data['Date'] = pd.to_datetime(option_data['Date'])
    
    # Group by date and calculate the mean of the features for each day
    time_series_data = option_data.groupby('Date').agg({
        'strike': 'mean',
        'best_bid': 'mean',
        'best_offer': 'mean',
        'implied_volatility': 'mean',
        'delta': 'mean',
        'gamma': 'mean',
        'vega': 'mean'
    }).reset_index()

    return time_series_data

# Plot estimated vs actual stock prices over time
def plot_estimated_vs_actual_stock_price(stock_price, time_series_option_data, estimated_prices_num, estimated_prices_bo):
    dates = time_series_option_data['Date']

    plt.figure(figsize=(12, 6))
    # Plot actual stock price
    plt.plot(dates, stock_price, label='Actual Stock Price', color='black', linestyle='-', marker='o')

    # Plot estimated stock price (Numerical Optimization)
    plt.plot(dates, estimated_prices_num, label='Estimated Price (Numerical Optimization)', color='blue', linestyle='--', marker='x')

    # Plot estimated stock price (Bayesian Optimization)
    plt.plot(dates, estimated_prices_bo, label='Estimated Price (Bayesian Optimization)', color='red', linestyle='-.', marker='s')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Estimated Stock Price Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function to test stock price estimation with both methods
def main():
    stocks = ["A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT"]
    expiration_date_between = ["2019-02-22", "2019-03-29"]

    # Step 1: Fetch stock price on the expiration dates (from yfinance)
    # try:
    #     stock_price = [get_stock_price(stock, date) for date in expiration_date_between]
    #     print(f"Stock prices on {expiration_date_between}: {stock_price}")
    # except ValueError as e:
    #     print(e)
    #     return
    # ground_truth = ((stock_price[1]['High'] + stock_price[1]['Low']) / 2).iloc[0]
    # Step 2: Fetch option data using DoltHub SQL query
    for stock in stocks: 
        option_data = pd.read_csv(f"{stock}_option.csv")#fetch_option_data(stock, expiration_date_between)  #pd.read_csv("option.csv")  
        # Step 3: Transform the option data into time-series format
        time_series_option_data = transform_option_data(option_data)
        print("Transformed Time-Series Option Data:")
        print(time_series_option_data.head())

        # Step 4: Perform Bayesian update and Bayesian optimization to estimate stock price
        mu_prior = 100  # Prior belief about the stock price
        sigma_prior = 15  # Uncertainty in the prior belief (standard deviation)
        r = 0.01  # Risk-free rate

        estimated_prices_num = []
        estimated_prices_bo = []
        for i, row in time_series_option_data.iterrows():
            observed_price = row['best_bid']  # Use best bid as the observed price for simplicity
            strike_price = row['strike']
            volatility = row['implied_volatility']
            print(row)
            breakpoint()
            # expiration_date = row['']
            time_to_expiration = 8/360 #(expiration_date - current_date).days / 365  

            # Estimate stock price using Bayesian update (numerical optimization)
            mu_prior, sigma_prior = update_estimate_with_observations([observed_price], strike_price, time_to_expiration, r, volatility, mu_prior, sigma_prior)
            estimated_prices_num.append(mu_prior)

            # Estimate stock price using Bayesian optimization
            estimated_price_bo, bo_result = bayesian_optimize_stock_price(observed_price, strike_price, time_to_expiration, r, volatility, mu_prior, sigma_prior)
            estimated_prices_bo.append(estimated_price_bo)

            print(f"Numerical Optimization - Estimated stock price for option with strike {strike_price}: {mu_prior}")
            print(f"Bayesian Optimization - Estimated stock price for option with strike {strike_price}: {estimated_price_bo}")

        # Step 5: Plot estimated vs actual stock prices over time
        breakpoint()
        print('simple bayesian update comparison to ground truth: ', abs(estimated_prices_num[-1] - ground_truth))
        print('bayesian optimization comparison to ground truth:', abs(estimated_prices_bo[-1] - ground_truth))
        # plot_estimated_vs_actual_stock_price(stock_price, time_series_option_data, estimated_prices_num, estimated_prices_bo)

if __name__ == "__main__":
    main()