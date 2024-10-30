from datetime import timedelta
from dateutil import parser
import yfinance as yf
import torch
from torch.nn.utils.rnn import pad_sequence


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



# Custom collate function to pad both X and y
def collate_fn(batch):
    # Separate features (X) and labels (y)
    X_batch, y_batch = zip(*batch)

    # Convert list of NumPy arrays to tensors
    X_batch = [torch.tensor(x, dtype=torch.float32) for x in X_batch]
    y_batch = [torch.tensor(y, dtype=torch.float32) for y in y_batch]

    # Pad sequences in X_batch to the same length
    X_batch_padded = pad_sequence(X_batch, batch_first=True)  # Pads X to the longest sequence in the batch

    # Pad sequences in y_batch to the same length (if needed)
    y_batch_padded = pad_sequence(y_batch, batch_first=True)  # Pads y to the longest sequence in the batch

    return X_batch_padded, y_batch_padded