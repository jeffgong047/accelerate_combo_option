import yfinance as yf
import pandas as pd
import os 
# Fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Fetch options data from Yahoo Finance
def get_option_data(ticker, expiration_date):
    ticker_obj = yf.Ticker(ticker)
    option_chain = ticker_obj.option_chain(expiration_date)
    calls = option_chain.calls
    puts = option_chain.puts
    return calls, puts

# Transform options data into a time-series format (aggregate features)
def transform_option_data(option_df):
    option_df['date'] = pd.to_datetime(option_df['lastTradeDate'])
    time_series_option = option_df.groupby('date').agg({
        'strike': 'mean',
        'impliedVolatility': 'mean',
        'bid': 'mean',
        'ask': 'mean',
        'volume': 'sum'
    }).reset_index()
    return time_series_option

# Merge stock and option data into a single dataset
def merge_stock_and_option_data(stock_data, option_data):
    merged_data = pd.merge(stock_data, option_data, left_index=True, right_on='date', how='inner')
    return merged_data

# Main function to execute the pipeline
def main():
    input_dir = "../data/"
    price_date = "20190123"
    ticker = "AAPL"
    opt_stats_filename = os.path.join(input_dir, ticker+"_"+price_date+".xlsx")
    opt_df_original = pd.read_excel(opt_stats_filename)
    expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))
    start_date = "2019-01-01"
    end_date = "2019-12-31"
    expiration_date = "2019-03-15"
    
    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    print(stock_data)
    breakpoint()
    # Get options data for the specified expiration date
    calls, puts = get_option_data(ticker, expiration_date)
    
    # Transform the options data (e.g., for calls)
    time_series_calls = transform_option_data(calls)
    
    # Merge stock and option data into a single time-series dataset
    merged_data = merge_stock_and_option_data(stock_data, time_series_calls)
    
    print("Merged Time-Series Data:")
    print(merged_data.head())

# Run the main function
if __name__ == "__main__":
    main()
