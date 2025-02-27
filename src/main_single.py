import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from match_prediction import BiAttentionClassifier_single as BiAttentionClassifier,train_model, validate_model, test_model, test_model_online
import numpy as np
import pickle
from utils import collate_fn
import os 
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=4, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=8, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set size as a percentage of the entire data')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the transformer')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to train the model')
    parser.add_argument('--offset', type=bool, default=False, help='Whether to allow offset for liability in the optimization')
    parser.add_argument('--stock_num', type=int, default=1, help='Number of stocks in the market for making option orders')
    parser.add_argument('--tasks', type=str, default='batch_profit_evaluation', help='Tasks to perform')
    return parser.parse_args()

def data_preprocessing(args):
    if args.stock_num == 1:
        data_file = '/common/home/hg343/Research/accelerate_combo_option/data/training_data_frontier_bid_ask_12_6.pkl'#'single_frontier.pkl' 
        model_path = 'single_offset_1.pt'#'single_no_offset.pt'#'temp_single_model.pt' 
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data, model_path 



def main():
    args = parse_arguments()
    data, model_path = data_preprocessing(args)
    # data = pd.read_csv('/common/home/hg343/Research/accelerate_combo_option/data/training_data_frontier_bid_ask.pkl')
# #sanity check :
# #1. Check if dominated by has elements if and only if the option does not belongs to frontier
# #2. Check if each market has frontiers

    # missing_dominated = []
    # missing_frontiers = [] 
    # for index, df in enumerate(data):
    #     assert df.loc[df['belongs_to_frontier'] == 1, 'dominated_by'].apply(lambda x: len(x)).sum() == 0 
    #     if df.loc[:,'belongs_to_frontier'].sum() == 0:
    #         missing_frontiers.append(index)
    #     if not all(df.loc[df['belongs_to_frontier'] == 0, 'dominated_by'].apply(lambda x: len(x))  > 0):
    #         missing_dominated.append(index)
    # print(missing_dominated, len(missing_dominated)/len(data))
    # print(missing_frontiers, len(missing_frontiers)/len(data)) 
    num_of_asks = [ len(x[(x['transaction_type'] == 0)])  for idx, x in enumerate(data)]
    num_of_bids = [ len(x[(x['transaction_type'] == 1)])  for idx, x in enumerate(data)]
    num_of_asks_in_the_frontier = [ len(x[(x['transaction_type'] == 0) & (x['belongs_to_frontier'] == 1)])  for idx, x in enumerate(data)]
    num_of_bids_in_the_frontier = [ len(x[(x['transaction_type'] == 1) & (x['belongs_to_frontier'] == 1)])  for idx, x in enumerate(data)]
    print(f'Number of asks: {sum(num_of_asks)}, Number of asks in the frontier: {sum(num_of_asks_in_the_frontier)}')
    print(f'Number of bids: {sum(num_of_bids)}, Number of bids in the frontier: {sum(num_of_bids_in_the_frontier)}')
    X = [x[['C=Call, P=Put',
            'Strike Price of the Option Times 1000',
            'B/A_price',
            'transaction_type']].to_numpy(dtype = np.float32) for x in data]

    y = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in data]


    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=args.test_split)
    # Flatten y_test to ensure it's a single array
    # Calculate percentage of labels = 1 in the entire dataset
    y_all_temp = np.concatenate(y)  # Concatenate to create a single array for all data
    percentage_labels_1_all = sum(y_all_temp) / len(y_all_temp) * 100  # Calculate percentage of labels = 1
    print(f"Percentage of labels = 1 in AAPL market: {percentage_labels_1_all:.2f}%")



    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(1 - args.train_split))

    
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(len(y_test))
    test_loader_online = DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = list(zip(X_test, y_test)) #DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False, collate_fn = collate_fn)
# list(zip(X_test, y_test))




    #load model using model_path and if only performs training if the model_path does not exist  
    if not os.path.exists(model_path):
        model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,num_classes=2)
        print('No saved model found, performing training')
    else:   
        model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,num_classes=2)
        model.load_state_dict(torch.load(model_path))
        print('Saved Model found, performing testing')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # true_probability = sum([np.sum(y_np) for y_np in y]) / sum([y_np.size for y_np in y])
    # pos_weight = torch.tensor([true_probability])

    # Use nn.BCEWithLogitsLoss for binary classification
    # weight = torch.tensor([1-pos_weight,pos_weight]
    frontier_loss_fn = nn.CrossEntropyLoss().float() #nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Train the model 
    if not os.path.exists(model_path):
        for i in range(args.iterations):
            train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=args.epochs)
            validate_model(model, val_loader, frontier_loss_fn)
    # Validate the model

    test_model(model, test_loader)
    if not os.path.exists(model_path):
        torch.save(model.state_dict(),model_path)
    breakpoint()
    test_model_online(model, test_loader_online, offset = 1,features = ['C=Call, P=Put',
    'Strike Price of the Option Times 1000',
    'B/A_price',
    'transaction_type'],**args)
    # torch.save(model.state_dict(),model_path)

    # Add this after creating the DataLoader
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Model input_size: {args.input_size}")
        break



if __name__ == '__main__':
    main()