from datetime import timedelta
from dateutil import parser
import yfinance as yf

def get_stock_price(stock, date):
    ticker = stock
    expiration_date = parser.parse(str(date))

    # Download historical data for the stock using yfinance
    stock_data = yf.download(ticker, start=expiration_date, end=expiration_date + timedelta(days=1))

    # Display the stock price on the expiration date
    if not stock_data.empty:
        print(f"Stock price for {ticker} on {expiration_date.date()}:")
        print(stock_data[['Open', 'High', 'Low', 'Close']])
        return stock_data[['Open', 'High', 'Low', 'Close']]
    else:
        print(f"No data available for {ticker} on {expiration_date.date()}.")