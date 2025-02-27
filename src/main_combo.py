import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
# from match_prediction import BiAttentionClassifier_single as BiAttentionClassifier
from match_prediction import train_model, validate_model, test_model , BiAttentionClassifier, test_model_online,finetune_policy_head,evaluate_policy_head
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
import numpy as np
import pickle
from utils import collate_fn
import os 
from tqdm import tqdm 
import itertools
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=6, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0025, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set size as a percentage of the entire data')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--iterations', type=int, default=30, help='Number of iterations to train the model')
    parser.add_argument('--market_size', type=int, default=50, help='Market size')
    parser.add_argument('--offset', type=int, default=0, help='Offset for the model')
    parser.add_argument('--tasks', type=str, nargs='+', default=['next_order_frontier_prediction'], help='List of tasks to perform (space-separated)')# 'batch_profit_evaluation', 'next_order_frontier_prediction','price_quote_evaluation'
    args = parser.parse_args()
    print("Parsed arguments:", vars(args))
    return args

def add_noise_to_create_match(df, num_orders=5):
    """Add noise to selected orders to create matching opportunities"""
    modified_df = df.copy()
    
    # Randomly select orders to modify
    indices = np.random.choice(len(df), size=min(num_orders, len(df)), replace=False)
    
    for idx in indices:
        # Set bid_or_ask to -1 only for ask orders (transaction_type == 0)
        if modified_df.iloc[idx, df.columns.get_loc('transaction_type')] == 0:
            modified_df.iloc[idx, df.columns.get_loc('transaction_type')] = -1
            
        # Add noise to price using the sign of transaction_type
        modified_df.iloc[idx, df.columns.get_loc('B/A_price')] = (
            modified_df.iloc[idx, df.columns.get_loc('B/A_price')] + 
            modified_df.iloc[idx, df.columns.get_loc('B/A_price')] * 
            modified_df.iloc[idx, df.columns.get_loc('transaction_type')] * 
            np.random.normal(0, 1)
        )
        
        # Reset ask orders back to 0 (after using -1 for noise calculation)
        if modified_df.iloc[idx, df.columns.get_loc('transaction_type')] == -1:
            modified_df.iloc[idx, df.columns.get_loc('transaction_type')] = 0
    
    return modified_df

def main():
    args = parse_arguments()
    combination = 2
    selecte_combination_name = 'corrected'
    model_path = f'/common/home/hg343/Research/accelerate_combo_option/src/combo_2_L_not_zero_model_weight.pth'
    dir_path = f'/common/home/hg343/Research/accelerate_combo_option/data/combo_{combination}_frontier'
    data = []
    num_files = 0
    no_matched_comb2 = [] 
    BA_KO = [] 
    BO_KO = []  
    MCD_KO = []
    DIS_XOM = []
    BA_HD = []
    IBM_JPM = [] 
    BA_KO_MCD_PG = []
    KO_MCD = []
    no_matched_files = []
    ba_ko_files = []
    corrected_files = []
    DJI = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM']
    possible_combinations = list(itertools.combinations(DJI,2))
    un_used_combinations = []
    used_combinations = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        keyword  = 'corrected'
        if os.path.isfile(file_path) and file.startswith(keyword) and file.endswith('AXP_MSFT.pkl'): #file_path.endswith('50.pkl'):
            # 
            combination = (file.split('_')[-1][:-4] , file.split('_')[-2]) 
            # i want to check if combination is in possible_combinations, and notice combination is unordered
            if combination in possible_combinations or tuple(reversed(combination)) in possible_combinations and combination not in used_combinations:
                used_combinations.append(combination)
            num_files += 1 
            print(file_path)
            creation_time = os.path.getctime(file_path)
            with open(file_path, 'rb') as f:
                data_point = pickle.load(f)
                df = pd.DataFrame(data_point, columns=['option1', 'option2','C=Call, P=Put',
                'Strike Price of the Option Times 1000',
                'transaction_type', 'B/A_price',
                'belongs_to_frontier'])
            if file_path.endswith('no_matched.pkl'):
                no_matched_files.append(creation_time)
                MCD_KO.append(df)
            elif file_path.endswith('BA_KO.pkl'):
                ba_ko_files.append(creation_time)
                BA_KO.append(df)
            elif file_path.endswith('DIS_XOM.pkl'):
                DIS_XOM.append(df)
            elif file_path.endswith('BA_HD.pkl'):
                BA_HD.append(df)
            elif file_path.endswith('IBM_JPM.pkl'):
                IBM_JPM.append(df)  
            elif file_path.endswith('BA_KO_MCD_PG.pkl'):
                BA_KO_MCD_PG.append(df)
            elif file_path.endswith('50.pkl'):
                KO_MCD.append(df)
            elif file.startswith('corrected'):
                corrected_files.append(df)
    combination_dict = {'BA_HD': BA_HD, 'BA_KO': BA_KO, 'DIS_XOM': DIS_XOM, 'IBM_JPM': IBM_JPM, 'BA_KO_MCD_PG': BA_KO_MCD_PG, 'KO_MCD': KO_MCD, 'corrected': corrected_files}
    for combination in possible_combinations:
        if combination not in used_combinations and tuple(reversed(combination)) not in used_combinations:
            un_used_combinations.append(combination)
    print(len(used_combinations))
    print(len(un_used_combinations))
    print(used_combinations)
    print(un_used_combinations)
    for combination_name, combination in combination_dict.items():
        num_of_asks = [ len(x[(x['transaction_type'] == 0)])  for idx, x in enumerate(combination)]
        num_of_bids = [ len(x[(x['transaction_type'] == 1)])  for idx, x in enumerate(combination)]
        num_of_asks_in_the_frontier = [ len(x[(x['transaction_type'] == 0) & (x['belongs_to_frontier'] == 1)])  for idx, x in enumerate(combination)]
        num_of_bids_in_the_frontier = [ len(x[(x['transaction_type'] == 1) & (x['belongs_to_frontier'] == 1)])  for idx, x in enumerate(combination)]
        print(f'for selected combination {combination_name} Number of asks: {sum(num_of_asks)}, Number of asks in the frontier: {sum(num_of_asks_in_the_frontier)}')
        print(f'for selected combination {combination_name} Number of bids: {sum(num_of_bids)}, Number of bids in the frontier: {sum(num_of_bids_in_the_frontier)}')
    BO_KO = DIS_XOM
                # mean_belongs_to_frontier = df['belongs_to_frontier'].mean()
                # if mean_belongs_to_frontier < 0.71:
                #     data.extend([df] * 10)
                # if mean_belongs_to_frontier < 0.65:
                #     data.extend([df] * 40)
                # if mean_belongs_to_frontier < 0.55:
                #     data.extend([df] * 50)


    # Assert that all no_matched files were created earlier than BA_KO files
    for ba_ko_time in ba_ko_files:
        assert all(ba_ko_time > no_matched_time for no_matched_time in no_matched_files), \
            "A BA_KO file was created before a no_matched file."
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
    selecte_combination = combination_dict[selecte_combination_name]

    # Add noisy versions of markets that create matches for training
    noisy_markets = []
    for index, market in enumerate(selecte_combination):
        if index > 1000:
            break
        noisy_market = add_noise_to_create_match(market)
        # Verify if the market now has matches
        buy_book = noisy_market[noisy_market['transaction_type'] == 1]
        sell_book = noisy_market[noisy_market['transaction_type'] == 0]
        # if synthetic_combo_match_mip(buy_book, sell_book)[3] > 0:  # Check if profit > 0
        noisy_markets.append(noisy_market)
    
    # Create separate noisy markets for evaluation
    eval_noisy_markets = []
    for index, market in enumerate(selecte_combination):
        if index <= 400 or index > 500:  # Skip the first 200 (used for training) and limit to 200 evaluation markets
            continue
        eval_noisy_market = add_noise_to_create_match(market)
        buy_book = eval_noisy_market[eval_noisy_market['transaction_type'] == 1]
        sell_book = eval_noisy_market[eval_noisy_market['transaction_type'] == 0]
        eval_noisy_markets.append(eval_noisy_market)
    
    print(f"Added {len(noisy_markets)} matching markets for training through noise addition")
    print(f"Added {len(eval_noisy_markets)} matching markets for evaluation through noise addition")
    features_order = ['option1','option2' ,'C=Call, P=Put',
            'Strike Price of the Option Times 1000',
            'B/A_price','transaction_type']
    selecte_combination_features = [x[features_order].to_numpy(dtype = np.float32) for x in selecte_combination]

    selecte_combination_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in selecte_combination]

    # X = [x[['C=Call, P=Put',
    #         'Strike Price of the Option Times 1000',
    #         'B/A_price',
    #         'transaction_type']].to_numpy(dtype = np.float32) for x in data]
    # y = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in data]

    selecte_combination_X_train_val, selecte_combination_X_test, selecte_combination_y_train_val, selecte_combination_y_test = train_test_split(selecte_combination_features, selecte_combination_labels, test_size=args.test_split)
    # Flatten y_test to ensure it's a single array
    # Calculate percentage of labels = 1 in the entire data
    y_all_temp = np.concatenate(selecte_combination_labels)  # Concatenate to create a single array for all data
    percentage_labels_1_all = sum(y_all_temp) / len(y_all_temp) * 100  # Calculate percentage of labels = 1
    print(f"Percentage of labels = 1 in {selecte_combination_name} market: {percentage_labels_1_all:.2f}%")

    # y_all_temp_BO_KO = np.concatenate(BO_KO_labels)  # Concatenate to create a single array for all BO_KO data
    # percentage_labels_1_BO_KO = sum(y_all_temp_BO_KO) / len(y_all_temp_BO_KO) * 100  # Calculate percentage of labels = 1 for BO_KO
    # print(f"Percentage of labels = 1 in BO_KO market: {percentage_labels_1_BO_KO:.2f}%")


    # Split BO_KO data into 20% for training and 80% for testing
    
    # Split MCD_KO training data
    X_train, X_val, y_train, y_val = train_test_split(selecte_combination_X_train_val, selecte_combination_y_train_val, test_size=(1 - args.train_split))
    
    # # Combine MCD_KO and BO_KO training data
    # X_train = X_train + BK_X_train
    # y_train = y_train + BK_y_train
    # Create data loaders
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = list(zip(selecte_combination_X_test, selecte_combination_y_test))
    test_loader_online = DataLoader(list(zip(selecte_combination_X_test, selecte_combination_y_test)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # test_generalization_loader = list(zip(BK_X_test, BK_y_test))

#  list(zip(X_test, y_test))
    model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,num_layers=5, num_classes=2) #bidirectional=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # true_probability = sum([np.sum(y_np) for y_np in y]) / sum([y_np.size for y_np in y])
    # pos_weight = torch.tensor([true_probability])

    # Use nn.BCEWithLogitsLoss for binary classification
    # weight = torch.tensor([1-pos_weight,pos_weight]
    frontier_loss_fn = nn.CrossEntropyLoss().float() #nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Train the model
    if not os.path.exists(model_path):
        for i in tqdm(range(args.iterations),desc='Training'):
            train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=args.epochs)
            validate_model(model, val_loader, frontier_loss_fn)

    # Test before finetuning
    print("Performance before finetuning:")
    test_model(model, test_loader)
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)

    # test_model_online(model, test_loader_online, features = ['option1','option2' ,'C=Call, P=Put',
    # 'Strike Price of the Option Times 1000',
    # 'B/A_price',
    # 'transaction_type'],**vars(args))

    # test_model(model, test_generalization_loader)

    # Finetuning phase
    #we use policy head to fine tune the model to direclty predict orders that matches. we also put punishment 
    # on total number of option orders that are involved.
    #add noise to train loader such that the orders in the train loader 
    # evaluate_policy_head(model, test_loader, reward_fn = synthetic_combo_match_mip, features_order = features_order, **vars(args))
    # breakpoint()
    # After creating noisy_markets and eval_noisy_markets, convert them to the format needed for training and evaluation

    # For training (noisy_markets)
    noisy_features = [x[features_order].to_numpy(dtype=np.float32) for x in noisy_markets]
    noisy_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in noisy_markets]

    # Create a DataLoader with the noisy markets for training
    noisy_loader = DataLoader(
        list(zip(noisy_features, noisy_labels)), 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # For evaluation (eval_noisy_markets)
    eval_noisy_features = [x[features_order].to_numpy(dtype=np.float32) for x in eval_noisy_markets]
    eval_noisy_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in eval_noisy_markets]

    # Create a list for evaluation (since evaluate_policy_head can handle lists)
    eval_noisy_data = list(zip(eval_noisy_features, eval_noisy_labels))



    # Use the noisy_loader for finetuning instead of train_loader
    fintuned_model = finetune_policy_head(
        model, 
        noisy_loader,  # Use noisy_loader instead of train_loader
        optimizer, 
        reward_fn=synthetic_combo_match_mip, 
        features=features_order,  # Make sure to use features=features_order
        **vars(args)
    )

    # Use eval_noisy_data for evaluation instead of test_loader
    evaluate_policy_head(
        fintuned_model, 
        eval_noisy_data,  # Use eval_noisy_data instead of test_loader
        reward_fn=synthetic_combo_match_mip, 
        features_order=features_order,
        **vars(args)
    )
    breakpoint()
    #evaluate the model on the eval_noisy_markets
    
    #test_matching_profit
    test_model(model, test_loader, task = 'matching_profit_evaluation')
   # Finetuning phase
    # Create a separate dataloader with only BK_X_train data
    # finetune_loader = DataLoader(list(zip(IBM_JPM_X_train_val, IBM_JPM_y_train_val)), 
    #                             batch_size=args.batch_size, 
    #                             shuffle=True, 
    #                             collate_fn=collate_fn)
    
    # Reduce learning rate for finetuning
    finetune_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate * 0.1)

    print("Starting finetuning phase...")
    for i in range(100):  # Fewer epochs for finetuning
        train_model(model, finetune_loader, finetune_optimizer, frontier_loss_fn, epochs=args.epochs)
        validate_model(model, val_loader, frontier_loss_fn)

    # Test after finetuning
    print("Performance after finetuning:")
    test_model(model, test_loader)
    # test_model(model, test_generalization_loader)

    # Save the finetuned model
    torch.save(model.state_dict(), model_path.replace('.pt', '_finetuned.pt'))

    # Add this after creating the DataLoader
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Model input_size: {args.input_size}")
        break



if __name__ == '__main__':
    main()