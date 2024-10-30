import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from match_prediction import BiAttentionClassifier,train_model, validate_model, test_model
import numpy as np
import pickle
from utils import collate_fn



def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=4, help='Number of input features (e.g. bid/ask prices)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test set size as a percentage of the entire data')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Assuming your data is loaded and preprocessed here
    # X = bid_ask_prices_features, y = frontier_targets
    # Split the data into training, validation, and test sets
    # data = pd.read_csv('training_data.csv')
    with open('training_data_with_buy_sell.pkl','rb') as f:
        data = pickle.load(f)
    X = [x[['C=Call, P=Put','Strike Price of the Option Times 1000','Highest Closing Bid Across All Exchanges','Lowest  Closing Ask Across All Exchanges']].to_numpy(dtype = np.float32) for x in data]
    y = [np.nan_to_num(y['belongs_to_frontier'].to_numpy(dtype = np.float32), nan=0) for y in data]
    X_padded, y_padded = collate_fn(list(zip(X,y)))
    mask = X_padded == 2
    selected_results = torch.masked_select(X_padded, mask)
    print(selected_results)
    breakpoint()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(1 - args.train_split))

    # Create DataLoader objects for train, validation, and test
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model, optimizer, and loss function
    model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,  num_classes=3, bidirectional=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    true_probability = sum([np.sum(y_np) for y_np in y]) / sum([y_np.size for y_np in y])  # Proportion of positive class
    # Calculate pos_weight: This is the weight for the positive class (class 1)
    pos_weight = torch.tensor([true_probability])

    # Use nn.BCEWithLogitsLoss for binary classification
    # weight = torch.tensor([1-pos_weight,pos_weight]
    frontier_loss_fn = nn.CrossEntropyLoss().float() #nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Train the model
    for i in range(200):
        train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=args.epochs)

    # Validate the model
        validate_model(model, val_loader, frontier_loss_fn)

    # Test the model
    test_model(model, test_loader)
    model_path = f'frontier_option_classifier_buy.pt'
    torch.save(model.state_dict(),model_path)

if __name__ == '__main__':
    main()