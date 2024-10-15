import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from match_prediction import OptionOrderDNN,train_model, validate_model, test_model
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=35, help='Number of input features (e.g. bid/ask prices)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test set size as a percentage of the entire data')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Assuming your data is loaded and preprocessed here
    # X = bid_ask_prices_features, y = frontier_targets
    # Split the data into training, validation, and test sets
    data = pd.read_csv('training_data.csv')
    X = data[['Stock', 'Expiration Date', 'Strike Price', 'Order Type', 'Value']]
    y = data['Frontier Label']
    X = pd.get_dummies(X, columns=['Stock', 'Order Type'])
    X = X.to_numpy(dtype = np.float32)
    y = y.to_numpy(dtype = np.float32)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(1 - args.train_split))

    # Create DataLoader objects for train, validation, and test
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False)

    # Initialize the model, optimizer, and loss function
    model = OptionOrderDNN(input_size=args.input_size, hidden_size=args.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    frontier_loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for classification

    # Train the model
    for i in range(10):
        train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=args.epochs)

    # Validate the model
        validate_model(model, val_loader, frontier_loss_fn)

    # Test the model
    test_model(model, test_loader)

if __name__ == '__main__':
    main()