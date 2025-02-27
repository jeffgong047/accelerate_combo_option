import time
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from utils import Mechanism_solver_single, Market
from epislon_priceQuote_frontierGeneration import financial_option_market
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
from combo_stock_frontier_data_preprocessor import synthetic_combo_frontier_generation
from copy import deepcopy
import matplotlib.pyplot as plt
from multiprocessing import Pool
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from torch import nn
# Training loop without stock targets


def verify_market_constraints(market, strikes):
    """Verify that all constraints are satisfied for given strikes"""
    opt = market.get_market_data_order_format().to_numpy()
    for strike in strikes:
        bid_sum = sum(max(opt[i,0]*(strike-opt[i,1]), 0) 
                     for i in range(len(opt)) if opt[i,3] == 1)
        ask_sum = sum(max(opt[i,0]*(strike-opt[i,1]), 0) 
                     for i in range(len(opt)) if opt[i,3] == 0)
        if bid_sum > ask_sum:
            return False, strike
    return True, None



def train_model(model, train_loader,optimizer ,frontier_loss_fn,epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_frontier_loss = 0

        for batch in train_loader:
            # Unpack the data; only bid_ask_prices and frontier_targets are provided
            bid_ask_prices, frontier_targets = batch
            # Forward pass
            frontier_pred = model(bid_ask_prices).float()
            # Frontier classification loss
            frontier_loss = frontier_loss_fn(frontier_pred.view(-1,frontier_pred.size(-1)), frontier_targets.view(-1).long())

            # Total loss (can be weighted if necessary)
            total_loss = frontier_loss  # We focus only on frontier prediction

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_frontier_loss += frontier_loss.item()

        print(f"Epoch {epoch+1}/{epochs},Frontier Loss: {total_frontier_loss}")

# Assuming you have a DataLoader that provides batches of option orders
# train_loader = your_data_loader
# train_model(model, train_loader, epochs=20)



def validate_model(model, val_loader, frontier_loss_fn):
    model.eval()
    total_frontier_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            bid_ask_prices, frontier_targets = batch
            frontier_pred = model(bid_ask_prices).float()
            frontier_loss = frontier_loss_fn(frontier_pred.view(-1, frontier_pred.size(-1)), frontier_targets.view(-1).long())
            total_frontier_loss += frontier_loss.item()
    print(f"Validation Loss: {total_frontier_loss}")


def run_matching_with_timeout(reward_fn, buy_book, sell_book, debug=0):
    try:
        return reward_fn(buy_book, sell_book, debug=debug)
    except Exception as e:
        print(f"Process error: {str(e)}")
        return None

def finetune_policy_head(model, train_loader, optimizer, reward_fn, epochs=4, features=['option1', 'option2','C=Call, P=Put',
                'Strike Price of the Option Times 1000',
                'B/A_price','transaction_type'], reward_weight= 1, penalty_weight=100, **args):
    """
    Finetune the policy head of the model to maximize profit while minimizing the number of selected orders.
    
    Args:
        model: The model to finetune
        train_loader: DataLoader for training data
        optimizer: Optimizer to use
        reward_fn: Function to compute reward
        epochs: Number of epochs to train
        features: List of feature names
        reward_weight: Weight for the reward loss
        penalty_weight: Weight for the penalty on number of selected orders
    """
    model.train()
    device = next(model.parameters()).device
    
    # Initialize metrics
    total_loss = 0
    total_profit = 0
    total_selected = 0
    
    # Initialize history lists to track metrics
    profit_ratio_history = []
    selection_ratio_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_profit = 0
        total_selected = 0
        
        for batch_idx, (bid_ask_prices, labels) in enumerate(train_loader):
            if isinstance(bid_ask_prices, list):
                bid_ask_prices = bid_ask_prices[0]
            
            if isinstance(labels, list):
                labels = labels[0]
            
            bid_ask_prices = bid_ask_prices.to(device)
            labels = labels.to(device)
            
            # Forward pass
            policy_pred = model(bid_ask_prices)
            
            # Get predicted labels (1 for selected, 0 for not selected)
            predicted_probs = torch.softmax(policy_pred, dim=-1)[:, :, 1]
            predicted_labels = (predicted_probs > 0.5).float()
            
            # Initialize lists to store profits and number of selected orders
            profits_list = []
            num_selected_list = []
            total_profit_list = []
            
            # Process each market in the batch
            for i in range(bid_ask_prices.size(0)):
                market_data = bid_ask_prices[i]
                selected = predicted_labels[i]
                
                # Count non-padding elements (actual orders)
                mask = (market_data.sum(dim=1) != 0).float()
                selected = selected * mask  # Zero out selections for padding
                
                # Count number of selected orders
                num_selected = selected.sum().item()
                num_selected_list.append(num_selected)
                
                # Skip if no orders are selected
                if num_selected == 0:
                    profits_list.append(torch.tensor(0.0, device=device, requires_grad=False))
                    total_profit_list.append(torch.tensor(0.0, device=device, requires_grad=False))
                    continue
                
                # Create dataframes for buy and sell books
                df = pd.DataFrame(market_data.detach().cpu().numpy(), columns=features)
                df['selected'] = selected.detach().cpu().numpy()
                
                # Filter selected orders
                selected_df = df[df['selected'] == 1]
                
                # Separate buy and sell books
                buy_book = selected_df[selected_df['transaction_type'] == 1]
                sell_book = selected_df[selected_df['transaction_type'] == 0]
                
                # Run matching algorithm with timeout
                try:
                    _, _, profit, _, _ = run_matching_with_timeout(reward_fn, buy_book, sell_book)
                    profits_list.append(torch.tensor(profit, device=device, requires_grad=False))
                    
                    # Calculate total possible profit (using all orders)
                    all_buy_book = df[df['transaction_type'] == 1]
                    all_sell_book = df[df['transaction_type'] == 0]
                    _, _, total_profit, _, _ = run_matching_with_timeout(reward_fn, all_buy_book, all_sell_book)
                    total_profit_list.append(torch.tensor(total_profit, device=device, requires_grad=False))
                    
                except Exception as e:
                    print(f"Error in matching: {e}")
                    profits_list.append(torch.tensor(0.0, device=device, requires_grad=False))
                    total_profit_list.append(torch.tensor(0.0, device=device, requires_grad=False))
            
            # Convert lists to tensors
            profits = torch.stack(profits_list)
            profit_values = profits.detach().cpu().numpy()
            
            # Convert total profit list to tensor
            total_profits = torch.tensor(total_profit_list, device=device, requires_grad=False)
            total_profit_values = total_profits.detach().cpu().numpy()
            
            # Calculate reward loss (negative because we want to maximize profit)
            reward_loss = -reward_weight * profits.mean()
            
            # Calculate policy loss (cross entropy between predicted and actual labels)
            policy_loss = nn.CrossEntropyLoss()(
                policy_pred.view(-1, policy_pred.size(-1)),
                predicted_labels.view(-1).detach().long()
            )
            
            # Combined loss
            total_batch_loss = reward_loss + policy_loss
            
            # Print components for debugging
            print(f"Reward loss: {reward_loss.item():.4f}, Policy loss: {policy_loss.item():.4f}")
            print(f"Avg profit: {profit_values.mean():.4f}, Avg selected: {sum(num_selected_list)/len(num_selected_list):.2f}")
            
            # Calculate profit ratio and selection ratio
            batch_profit_sum = profit_values.sum()
            batch_total_profit_sum = total_profit_values.sum()
            
            # Count total number of orders in the batch
            batch_total_orders = sum(len(market_data) for market_data in bid_ask_prices)
            batch_selected_orders = sum(num_selected_list)
            
            # Calculate and print ratios
            profit_ratio = batch_profit_sum / batch_total_profit_sum if batch_total_profit_sum != 0 else 0
            selection_ratio = batch_selected_orders / batch_total_orders if batch_total_orders > 0 else 0
            
            # Add current ratios to history
            profit_ratio_history.append(profit_ratio)
            selection_ratio_history.append(selection_ratio)
            
            # Print current ratios and history
            print(f"Profit ratio: {profit_ratio:.4f}, Selection ratio: {selection_ratio:.4f}")
            print(f"Profit ratio history: {[f'{pr:.4f}' for pr in profit_ratio_history]}")
            print(f"Selection ratio history: {[f'{sr:.4f}' for sr in selection_ratio_history]}")
            
            # Optimization step
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += total_batch_loss.item()
            total_profit += profits.sum().item()
            total_selected += sum(num_selected_list)
        
        avg_selected = total_selected / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss:.2f} | "
              f"Avg Profit: {total_profit/len(train_loader) if len(train_loader) > 0 else 0:.2f} | "
              f"Avg Selected: {avg_selected:.1f}")
    
    # Print final history at the end of training
    print("\nFinal Metrics History:")
    print(f"Profit ratio history: {[f'{pr:.4f}' for pr in profit_ratio_history]}")
    print(f"Selection ratio history: {[f'{sr:.4f}' for sr in selection_ratio_history]}")
    
    return model

def evaluate_policy_head(model, test_loader, reward_fn, features_order, **args):
    """Evaluate finetuned policy head on unseen test data"""
    model.eval()
    total_profit = 0
    total_selected = 0
    baseline_profit = 0
    
    with torch.no_grad():
        # Check if test_loader is a DataLoader or a list
        if hasattr(test_loader, 'dataset'):
            # It's a DataLoader
            dataset_size = len(test_loader.dataset)
            is_dataloader = True
        else:
            # It's a list
            dataset_size = len(test_loader)
            is_dataloader = False
            
        # Iterate through the data
        for batch in test_loader:
            if is_dataloader:
                bid_ask_prices, _ = batch
            else:
                # If it's a list, each item is already a (bid_ask_prices, _) tuple
                bid_ask_prices, _ = batch
            
            # Convert to tensor if it's a numpy array
            if isinstance(bid_ask_prices, np.ndarray):
                bid_ask_prices = torch.tensor(bid_ask_prices, dtype=torch.float32)
            
            # Ensure proper dimensions - add batch dimension if needed
            if bid_ask_prices.dim() == 2:
                bid_ask_prices = bid_ask_prices.unsqueeze(0)
                
            # Get policy predictions
            policy_pred = model(bid_ask_prices, use_policy_head=True)
            predicted_labels = policy_pred.argmax(dim=-1)
            selected_mask = predicted_labels == 1
            
            # Process each market in batch
            for i in range(bid_ask_prices.size(0)):
                filtered_orders = bid_ask_prices[i][selected_mask[i]]
                all_orders = bid_ask_prices[i]
                
                if len(filtered_orders) > 0:
                    # Convert to numpy and create markets
                    filtered_orders_np = filtered_orders.detach().cpu().numpy()
                    all_orders_np = all_orders.detach().cpu().numpy()
                    
                    try:
                        # Create markets and compute profits
                        filtered_orders_df = pd.DataFrame(filtered_orders_np, columns=features_order)
                        market = Market(filtered_orders_df, input_format=None)
                        filtered_buy, filtered_sell = market.separate_buy_sell()
                        
                        all_orders_df = pd.DataFrame(all_orders_np, columns=features_order)
                        all_orders_market = Market(all_orders_df, input_format=None)
                        all_buy, all_sell = all_orders_market.separate_buy_sell()
                        
                        # Check if we have both buy and sell orders
                        if len(filtered_buy) > 0 and len(filtered_sell) > 0:
                            # Compute profits using pool with timeout
                            with pool_context(processes=1) as pool:
                                try:
                                    # Run filtered market matching with timeout
                                    async_result = pool.apply_async(run_matching_with_timeout, 
                                                                  (reward_fn,filtered_buy, filtered_sell))
                                    _, _, policy_profit, _, _ = async_result.get(timeout=60)
                                    print('policy_profit',policy_profit)
                                    # Run all orders matching with timeout
                                    async_result = pool.apply_async(run_matching_with_timeout, 
                                                                  (reward_fn,all_buy, all_sell))
                                    _, _, all_profit, _, _ = async_result.get(timeout=60)
                                    print('all_profit',all_profit)
                                    total_profit += policy_profit
                                    baseline_profit += all_profit
                                    total_selected += selected_mask[i].sum().item()
                                    
                                except TimeoutError:
                                    print("Matching operation timed out after 60 seconds")
                                    continue
                                except Exception as e:
                                    print(f"Error in matching: {str(e)}")
                                    continue
                    except Exception as e:
                        print(f"Error processing market: {e}")
                        continue
    
    avg_selected = total_selected / dataset_size if dataset_size > 0 else 0
    profit_ratio = total_profit / baseline_profit if baseline_profit != 0 else 0
    
    print(f"Policy Evaluation Results:")
    print(f"Total Profit: {total_profit:.2f} | Baseline Profit: {baseline_profit:.2f}")
    print(f"Profit Ratio: {profit_ratio:.2f} | Avg Selected: {avg_selected:.1f}")
    
    return total_profit, baseline_profit, profit_ratio, avg_selected



# def compute_reward(predicted_frontier, bid_ask_prices):
#     """
#     Compute the reward for the policy predictions.
#     Returns: tensor of shape [batch_size] containing rewards
#     """
#     rewards = []
#     for i in range(len(bid_ask_prices)):
#         # Create market from single prediction
#         market_data = bid_ask_prices[i][predicted_frontier[i] == 1]
#         if len(market_data) > 0:
#             market = Market(market_data)
#             profit = Mechanism_solver_single(market)
#             rewards.append(profit)
#         else:
#             rewards.append(0.0)
    
#     return torch.tensor(rewards, device=predicted_frontier.device)


def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_loader:
            bid_ask_prices, frontier_targets = batch  # Unpack batch data
            bid_ask_prices, frontier_targets = torch.tensor(bid_ask_prices).unsqueeze(0), torch.tensor(frontier_targets).unsqueeze(0)
            # Forward pass: Get logits from the model
            frontier_pred = model(bid_ask_prices)  # Output shape: [batch_size, seq_len, num_classes]

            # Convert logits to predicted class (0 or 1) using argmax
            predicted = frontier_pred.argmax(dim=-1).cpu().numpy()  # Get the class with the highest logit
            targets = frontier_targets.cpu().numpy()  # Convert targets to numpy for metrics

            # Accumulate predictions and true labels
            all_predictions.append(predicted.flatten())  # Flatten in case of sequence
            all_targets.append(targets.flatten())

    # Convert accumulated lists to numpy arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Compute precision, recall, f1 score
    precision = precision_score(all_targets, all_predictions, average='binary')
    recall = recall_score(all_targets, all_predictions, average='binary')
    f1 = f1_score(all_targets, all_predictions, average='binary')

    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # Print metrics
    accuracy = np.mean(all_predictions == all_targets)  # Calculate accuracy
    frontier_percentage = (all_targets.sum() / len(all_targets)) * 100
    print(f"Frontier percentage in the testing dataset: {frontier_percentage:.2f}%")
    print(f"Accuracy: {accuracy:.4f}")  # Print accuracy
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)



    

@contextmanager
def pool_context(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()



def test_model_online(model, test_loader, single_stock = False,offset = 0,**args):
    available_tasks = ['batch_profit_evaluation', 'next_order_frontier_prediction','price_quote_evaluation']
    tasks = args['tasks']
    tasks = {task: task in tasks for task in available_tasks}
    print('performing tasks',tasks)
    model.eval()
    DNN_inference_time_total = 0
    DNN_inference_profits = []
    solver_profits = []
    all_orders_profits = []
    solver_time_per_market = []
    predicted_markets_info = []
    price_quote_all_markets = []
    price_quote_solver_markets = []
    price_quote_DNN_inference_markets = []
    frontier_probability_list = []
    # Initialize counters before the loop
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative
    total_prediction = 0
    DNN_inference_start = time.time()
    #only compute the inference time
    for batch in test_loader:
        bid_ask_prices, frontier_targets = batch
        
        # Forward pass with original tensors
        frontier_pred = model(bid_ask_prices)
        predicted = frontier_pred.argmax(dim=-1)
    DNN_inference_end = time.time()
    DNN_inference_time_total += DNN_inference_end - DNN_inference_start

    #compute 
    for batch in test_loader:
        bid_ask_prices, frontier_targets = batch
        
        # Forward pass with original tensors
        frontier_pred = model(bid_ask_prices)
        predicted = frontier_pred.argmax(dim=-1)
        
        # Now process each market in the batch
        for i in range(len(bid_ask_prices)):
            # Get actual sequence length for this market (excluding padding)
            length = (bid_ask_prices[i] != 0).any(dim=-1).sum()
            
            # Get valid predictions and targets for this market
            valid_predicted = predicted[i, :length]
            valid_targets = frontier_targets[i, :length]
            
            # Create masks for frontier points
            predicted_mask = valid_predicted == 0
            targets_mask = valid_targets == 0
            # Get indices where both masks are True
            option_indices = torch.where(predicted_mask & targets_mask)  # [0] to get just the indices array
            #prepare a common indexing for bid_ask_prices, valid_predicted, valid_targets for pandas dataframe 
            index = np.arange(0, bid_ask_prices[i].shape[0], 1)
            if len(option_indices) == 1 and len(option_indices[0]) > 1:
                predicted_markets_info.append([
                    pd.DataFrame(bid_ask_prices[i, :length].numpy(), columns = args['features'], index = index[:length]),  # Only keep non-padded data
                    pd.DataFrame(valid_predicted.numpy(), columns = ['prediction'], index = index[:length]),
                    pd.DataFrame(valid_targets.numpy(), columns = ['target'], index = index[:length]),
                    option_indices,
                    length
                ])
    
    #perform exact solver calculation 
    for batch in test_loader:
        bid_ask_prices, frontier_targets = batch
        for market in bid_ask_prices:
            start_solver = time.time()
            length = (market!= 0).any(dim=-1).sum()
            # Ensure bid_ask_prices is 2D before creating DataFrame
            # Assuming you want to use the first 100 rows and all columns

            
            # Create DataFrame from 2D tensor
            market = pd.DataFrame(market[:length].numpy(), columns=args['features'])
            # if single_stock: 
            #     solved_frontier = frontier_solver(Market(market))
            # else:
            #     opt_buy_book , opt_sell_book  = market[market['transaction_type'] == 1], market[market['transaction_type'] == 0]
            #     if len(opt_buy_book) == 0 or len(opt_sell_book) == 0:
            #         print("Warning: Empty order book detected, skipping frontier generation")
            #         continue        
            #     frontier = synthetic_combo_frontier_generation(opt_buy_book.to_numpy(), opt_sell_book.to_numpy(), debug=0)
            end_solver = time.time()
            # breakpoint()
            # try:
            #     assert np.all(solved_frontier['belongs_to_frontier'] == frontier_targets[i][:length].numpy())
            # except:
            #     breakpoint()
            #     a=1
            #     b=2
            #     c=a+b
            solver_time_per_market.append(end_solver - start_solver)
    solver_time_total = sum(solver_time_per_market)
    #use mechanism solver to compute profit for each market 
    # i  = 0 
    for market, predicted, targets, option_indices, length in predicted_markets_info:

        i += 1 
        if i>100:
            break

        #debug frontier labels in the market. we feed the market data into synthetic_combo_frontier_generation,
        #and check if the labels are the same as the targets
        price_index = args['features'].index('B/A_price')
        bid_or_ask = args['features'].index('transaction_type')
        buy_book_index = market[market.iloc[:,bid_or_ask] == 1].index
        sell_book_index = market[market.iloc[:,bid_or_ask] == 0].index
        #compare whether frontier labels align with targets based on indexes
        if single_stock:
            if Mechanism_solver_single(Market(market))[0] == 1:
                breakpoint() #
        elif synthetic_combo_match_mip(market[market.iloc[:,bid_or_ask] == 1], market[market.iloc[:,bid_or_ask] == 0])[3] != 0:
                print('given market itself matches. It cant be like this')
                breakpoint()
        if tasks['batch_profit_evaluation']:
            new_coming_orders = deepcopy(market.loc[option_indices[0][:3]]) # Directly use option_indices
            new_coming_orders_index = new_coming_orders.index
            new_coming_orders = new_coming_orders.to_numpy()
            ask_mask = new_coming_orders[:,bid_or_ask] == 0 
            new_coming_orders[:,bid_or_ask][ask_mask] = -1
            new_coming_orders[:,price_index] = new_coming_orders[:,price_index] + new_coming_orders[:,price_index]* new_coming_orders[:,bid_or_ask]*np.random.normal(0, 1, size=new_coming_orders.shape[0])
            new_coming_orders[:,bid_or_ask][ask_mask] = 0 
            #get index number of those predicted[i] ==1
            predicted_options = deepcopy(market[predicted['prediction'] == 1])
            solver_options = deepcopy(market[targets['target'] == 1])
            test_market_DNN = pd.concat([pd.DataFrame(new_coming_orders, columns = args['features'], index = new_coming_orders_index), predicted_options], axis = 0)
            test_market_solver = pd.concat([pd.DataFrame(new_coming_orders, columns = args['features'], index = new_coming_orders_index), solver_options], axis = 0)
            test_all_orders = deepcopy(market)
            test_all_orders.loc[new_coming_orders_index] = deepcopy(new_coming_orders)
            test_all_orders_market = Market(test_all_orders, columns = args['features'], raw_data_format = None)
            test_DNN_inference_market = Market(test_market_DNN, columns = args['features'], raw_data_format = None)
            test_solver_market = Market(test_market_solver, columns = args['features'], raw_data_format = None)

            assert test_market_DNN.index.isin(test_all_orders_market.get_market_data_order_format().index).all(), "DNN market not subset of all orders"
            assert test_market_solver.index.isin(test_all_orders_market.get_market_data_order_format().index).all(), "Solver market not subset of all orders"
            assert test_market_DNN.loc[test_market_DNN.index].equals(test_all_orders_market.get_market_data_order_format().loc[test_market_DNN.index]), "Data mismatch in DNN market"
            assert test_market_solver.loc[test_market_solver.index].equals(test_all_orders_market.get_market_data_order_format().loc[test_market_solver.index]), "Data mismatch in solver market"
            #please add assertion to verify that strikes from all markets are the same. Strikes should be list and I just assume equivalence not w.r.t to order so you could order it
            print('solving for all orders')
            if single_stock:
                _, profit_all_orders, all_orders_matched_indices,_ = Mechanism_solver_single(test_all_orders_market, offset = offset)
                print('solving for solver market')
                solver_matched,profit_solver,solver_matched_indices,_ = Mechanism_solver_single(test_solver_market, offset = offset)
                print('solving for DNN inference market')
                DNN_matched, profit_DNN_inference,DNN_matched_indices,_ = Mechanism_solver_single(test_DNN_inference_market, offset = offset)
            else:
                test_solver_market_opt_buy_book , test_solver_market_opt_sell_book  = test_solver_market.separate_buy_sell()
                test_DNN_inference_market_opt_buy_book , test_DNN_inference_market_opt_sell_book  = test_DNN_inference_market.separate_buy_sell()
                if len(test_solver_market_opt_buy_book) == 0 or len(test_solver_market_opt_sell_book) == 0 or len(test_DNN_inference_market_opt_buy_book) == 0 or len(test_DNN_inference_market_opt_sell_book) == 0:
                    continue
                all_orders = pd.concat(
                    (
                        pd.DataFrame(new_coming_orders, columns=args['features'], index=new_coming_orders_index),
                        market
                    ), 
                    axis=0
                )
                #buy book contains bid orders; sell_book contains ask orders 
                all_orders_buy_book, all_orders_sell_book = Market(all_orders, columns=args['features'], raw_data_format=None).separate_buy_sell()
                with pool_context(processes=1) as pool:
                    try:
                        # Run all orders matching
                        async_result = pool.apply_async(run_matching_with_timeout, 
                                                      (all_orders_buy_book, all_orders_sell_book))
                        time_all_orders, _, profit_all_orders, _, matched_stock_all_orders = async_result.get(timeout=60)

                        # Run DNN inference matching
                        async_result = pool.apply_async(run_matching_with_timeout, 
                                                      (test_DNN_inference_market_opt_buy_book, test_DNN_inference_market_opt_sell_book))
                        time_DNN_inference, _, profit_DNN_inference, _, matched_stock_DNN_inference = async_result.get(timeout=60)

                        # Run solver matching
                        async_result = pool.apply_async(run_matching_with_timeout, 
                                                      (test_solver_market_opt_buy_book, test_solver_market_opt_sell_book))
                        time_solver, _, profit_solver, _, matched_stock_solver = async_result.get(timeout=60)

                    except TimeoutError:
                        print("Matching operation timed out after 60 seconds")
                        continue
                    except Exception as e:
                        print(f"Error in matching: {str(e)}")
                        continue
                print('All orders profits:', profit_all_orders)
                print('All orders Buy Book indices:', all_orders_buy_book.index.tolist())
                print('All orders Sell Book indices:', all_orders_sell_book.index.tolist())

                print('DNN inference profits:', profit_DNN_inference)
                print('DNN inference Buy Book indices:', test_DNN_inference_market_opt_buy_book.index.tolist())
                print('DNN inference Sell Book indices:', test_DNN_inference_market_opt_sell_book.index.tolist())

                print('Solver profits:', profit_solver)
                print('Solver Buy Book indices:', test_solver_market_opt_buy_book.index.tolist())
                print('Solver Sell Book indices:', test_solver_market_opt_sell_book.index.tolist())

                print('Matched stock all orders:', matched_stock_all_orders)
                print('Matched stock DNN inference:', matched_stock_DNN_inference)
                print('Matched stock solver:', matched_stock_solver)
        if tasks['next_order_frontier_prediction']:
            next_order_non_frontier = deepcopy(market.loc[option_indices[0][:1]])   # this order is not in the frontier 
            next_order_frontier = deepcopy(next_order_non_frontier)
            if single_stock:
                assert Mechanism_solver_single(Market(market))[0] == 0
            else:
                # Fix: Convert market data to torch tensor with consistent dtype before model inference
                market_data = torch.tensor(market.values, dtype=torch.float32)
                buy_mask = market_data[:, bid_or_ask] == 1
                sell_mask = market_data[:, bid_or_ask] == 0
                
                buy_book = market[market.iloc[:, bid_or_ask] == 1]
                sell_book = market[market.iloc[:, bid_or_ask] == 0]
                
                assert synthetic_combo_match_mip(buy_book, sell_book)[3] == 0

            sufficient = False 
            while not sufficient:
                next_order_frontier.iloc[:, bid_or_ask] = -1
                next_order_frontier.iloc[:, price_index] = (
                    next_order_frontier.iloc[:, price_index] + 
                    next_order_frontier.iloc[:, price_index] * 
                    next_order_frontier.iloc[:, bid_or_ask] * 
                    np.random.normal(0, 1, size=next_order_frontier.shape[0])
                )
                next_order_frontier.iloc[:, bid_or_ask] = 0
                
                new_market = pd.concat([market, next_order_frontier], axis=0)
                if single_stock:
                    sufficient = Mechanism_solver_single(Market(new_market))[0] == 1
                else:
                    buy_book = new_market[new_market.iloc[:, bid_or_ask] == 1]
                    sell_book = new_market[new_market.iloc[:, bid_or_ask] == 0]
                    sufficient = synthetic_combo_match_mip(buy_book, sell_book)[3] == 1
            #use DNN to infer whether the new coming order is in the frontier 
            input = pd.concat([market, next_order_frontier], axis = 0)
            input_tensor = torch.tensor(input.to_numpy(), dtype=torch.float32).unsqueeze(0)
            frontier_pred = model(input_tensor)
            # Apply softmax to get probability distribution
            probabilities = torch.nn.functional.softmax(frontier_pred, dim=-1)
            next_order_frontier_prob = probabilities.squeeze(0)  # Remove batch dimension
            # Get probability of being in frontier (class 1) for the last order
            frontier_probability = next_order_frontier_prob[-1, 1].item()
            
            # True label is 1 (frontier) in this case
            true_label = 1
            print('frontier probability',frontier_probability)
            predicted_label = 1 if frontier_probability > 0.5 else 0
            print('predicted label',predicted_label)
            frontier_probability_list.append(frontier_probability)
            if true_label == 1 and predicted_label == 1:
                tp += 1
            elif true_label == 1 and predicted_label == 0:
                fn += 1
            elif true_label == 0 and predicted_label == 1:
                fp += 1
            else:
                tn += 1
            
            total_prediction += 1
            print(f'Total predictions: {total_prediction}')
            print(f'True Positives: {tp}, False Positives: {fp}')
            print(f'True Negatives: {tn}, False Negatives: {fn}')
            print(f'Frontier probability: {frontier_probability}')
            
            # Calculate rates
            if total_prediction > 0:
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                print(f'True Positive Rate: {tpr:.3f}')
                print(f'True Negative Rate: {tnr:.3f}')
                print(f'False Positive Rate: {fpr:.3f}')
                print(f'False Negative Rate: {fnr:.3f}')
        if tasks['price_quote_evaluation']:
            #use DNN to predict frontier to quote the price
            new_coming_orders = deepcopy(market.loc[option_indices[0][:1]]) # Directly use option_indices
            new_coming_orders_index = new_coming_orders.index
            new_coming_orders = new_coming_orders.to_numpy()
            ask_mask = new_coming_orders[:,bid_or_ask] == 0 
            new_coming_orders[:,bid_or_ask][ask_mask] = -1
            new_coming_orders[:,price_index] = new_coming_orders[:,price_index] + new_coming_orders[:,price_index]* new_coming_orders[:,bid_or_ask]*np.random.normal(0, 1, size=new_coming_orders.shape[0])
            new_coming_orders[:,bid_or_ask][ask_mask] = 0 
            #get index number of those predicted[i] ==1
            predicted_options = deepcopy(market[predicted['prediction'] == 1])
            solver_options = deepcopy(market[targets['target'] == 1])
            test_market_DNN = pd.concat([pd.DataFrame(new_coming_orders, columns = args['features'], index = new_coming_orders_index), predicted_options], axis = 0)
            test_market_solver = pd.concat([pd.DataFrame(new_coming_orders, columns = args['features'], index = new_coming_orders_index), solver_options], axis = 0)
            test_all_orders = deepcopy(market)
            test_all_orders.loc[new_coming_orders_index] = deepcopy(new_coming_orders)
            test_all_orders_market = Market(test_all_orders, columns = args['features'], raw_data_format = None)
            test_DNN_inference_market = Market(test_market_DNN, columns = args['features'], raw_data_format = None)
            test_solver_market = Market(test_market_solver, columns = args['features'], raw_data_format = None)
            #use all options in the market to quote the price 
            all_orders_buy_book, all_orders_sell_book = Market(test_all_orders, columns=args['features'], raw_data_format=None).separate_buy_sell()
            price_quote = price_quote_combo(test_all_orders_market, new_coming_orders,debug=0) 
            #use solver to quote the price 
            price_quote_solver = price_quote_combo(test_solver_market, new_coming_orders, debug=0)
            #use DNN inference to quote the price 
            price_quote_DNN_inference = price_quote_combo(test_DNN_inference_market, new_coming_orders, debug=0)
            print(price_quote,price_quote_solver,price_quote_DNN_inference)
        

        if tasks['batch_profit_evaluation']:
            DNN_inference_profits.append(profit_DNN_inference)
            solver_profits.append(profit_solver)
            all_orders_profits.append(profit_all_orders)
        if tasks['price_quote_evaluation']:
            price_quote_all_markets.append(price_quote)
            price_quote_solver_markets.append(price_quote_solver)
            price_quote_DNN_inference_markets.append(price_quote_DNN_inference)

            # DNN_strikes = test_DNN_inference_market.get_strikes()
            # all_orders_strikes  = test_all_orders_market.get_strikes()
        # if round(profit_DNN_inference,2) > round(profit_all_orders,2):
        #     # Verify constraints for both markets
        #     all_strikes = sorted(set(test_all_orders_market.get_strikes() + test_DNN_inference_market.get_strikes()))
        #     dnn_valid, dnn_violating_strike = verify_market_constraints(test_DNN_inference_market, all_strikes)
        #     all_valid, all_violating_strike = verify_market_constraints(test_all_orders_market, all_strikes)
            
        #     print(f"DNN market constraints valid: {dnn_valid}")
        #     print(f"All orders market constraints valid: {all_valid}")
            
        #     if not dnn_valid:
        #         print(f"DNN market violates constraint at strike {dnn_violating_strike}")
        #     if not all_valid:
        #         print(f"All orders market violates constraint at strike {all_violating_strike}")

            #write me the code to index from all_orders that are not in the market of test_DNN_inference_market

            # i want to test strikes that is not in DNN market but in all orders market and see if constraint still hold for that point
            # DNN_strikes = test_DNN_inference_market.get_strikes()
            # all_orders_strikes  = test_all_orders_market.get_strikes()
            # #get those in all_orders_strikes but not in DNN strikes 
            # all_orders_strikes_not_in_DNN_strikes = [strike for strike in all_orders_strikes if strike not in DNN_strikes]
            # DNN_matched, profit_DNN_inference,DNN_matched_indices,_ = Mechanism_solver_single(test_DNN_inference_market, all_orders_strikes_not_in_DNN_strikes)
            # all_orders_indices = test_all_orders_market.get_market_data_order_format().index
            # all_orders_indices = test_all_orders_market.get_market_data_order_format().index
            # DNN_inference_market_indices = test_DNN_inference_market.get_market_data_order_format().index
            # all_orders_indices_not_in_DNN_inference_market = [idx for idx in all_orders_indices if idx not in DNN_inference_market_indices]
            # # print(all_orders_indices_not_in_DNN_inference_market)
            # orders_not_in_DNN_inference_market = test_all_orders_market.get_market_data_order_format().loc[all_orders_indices_not_in_DNN_inference_market]
            # test_all_orders_market.drop_index(all_orders_indices_not_in_DNN_inference_market)
            # print('removing orders not in DNN inference market from all orders marekt',Mechanism_solver_single(test_all_orders_market)[1])
            # #
            # breakpoint()
            # print('adding orders not in DNN inference market to DNN inference market',Mechanism_solver_single(Market(pd.concat([orders_not_in_DNN_inference_market, test_DNN_inference_market.get_market_data_order_format()], axis = 0)))[1])
            # try:
            #     assert sorted(test_DNN_inference_market.get_strikes()) == sorted(test_solver_market.get_strikes()) and sorted(test_DNN_inference_market.get_strikes()) == sorted(test_all_orders_market.get_strikes()), "Strikes mismatch in DNN and solver markets"
            # except:
            #     #i want you to print out those strikes the mismatch
            #     print('DNN inference market strikes',test_DNN_inference_market.get_strikes(),len(test_DNN_inference_market.get_strikes()))
            #     print('solver market strikes',test_solver_market.get_strikes(),len(test_solver_market.get_strikes()))
            #     print('all orders market strikes',test_all_orders_market.get_strikes(),len(test_all_orders_market.get_strikes()))
            #     breakpoint()
            #     a=1
            #     b=2
            #     c=a+b
            #     print('solving for all orders')
            
            # print("\nDetailed market comparison:")
            # debug_market_comparison(test_all_orders_market, test_DNN_inference_market)


    # print(f"DNN inference time total: {DNN_inference_time_total}")
    # print(f"Solver time per market: {solver_time_total}")
    # print(f"Profit of ground truth: {profit_all}")
    if tasks['next_order_frontier_prediction']:
        num_correct_prediction = tp + tn
        print(f"Accuracy of DNN inference: {num_correct_prediction/total_prediction}")
        breakpoint()
    if tasks['batch_profit_evaluation']:
        print(f"Profit of solver: {solver_profits}")
        print(f"Profit of DNN inference: {DNN_inference_profits}")
        print(f"Profit of all orders: {all_orders_profits}")
    #print out the proft time ratio of both DNN_inference and solver 
    # print(f"Profit time ratio of DNN inference: {sum(DNN_inference_profits)/DNN_inference_time_total}")
    # print(f"Profit time ratio of solver: {sum(solver_profits)/solver_time_total}")
        plt.figure(figsize=(12, 6))
        plt.plot(solver_profits, label="Profit of Solver", marker='o')
        plt.plot(DNN_inference_profits, label="Profit of DNN Inference", marker='x')
        plt.plot(all_orders_profits, label="Profit of All Orders", marker='s')
        # Customize plot
        plt.title("Profit Comparison", fontsize=14)
        plt.xlabel("Order Index", fontsize=12)
        plt.ylabel("Profit", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/common/home/hg343/Research/accelerate_combo_option/visualizations/combo_{single_stock}_profit_comparison_offset{offset}.png')
    # i want to add a figure that could plot all_orders_profits - solver_profits against all_orders_profit - DNN_inference_profits
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(all_orders_profits) - np.array(solver_profits), label="Profit of All Orders - Solver", marker='o')
        plt.plot(np.array(all_orders_profits) - np.array(DNN_inference_profits), label="Profit of All Orders - DNN Inference", marker='x')
        plt.title("Profit Difference Comparison", fontsize=14)
        plt.xlabel("Order Index", fontsize=12)
        plt.ylabel("Profit Difference", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/common/home/hg343/Research/accelerate_combo_option/visualizations/combo_{single_stock}_profit_difference_comparison_offset{offset}.png')
        #calculate the variance of two arrays to be ploted 
        print('percentage of kept profits DNN_inference', sum(DNN_inference_profits)/sum(all_orders_profits))
        print('percentage of kept profits solver', sum(solver_profits)/sum(all_orders_profits))
        #calculate the medium absolute deviation instead
        print('medium absolute deviation of all_orders_profits - solver_profits',np.median(np.abs(np.array(all_orders_profits) - np.array(solver_profits))))
        print('medium absolute deviation of all_orders_profits - DNN_inference_profits',np.median(np.abs(np.array(all_orders_profits) - np.array(DNN_inference_profits))))
        # calculate the variance 
        print('variance of all_orders_profits - solver_profits',np.var(np.array(all_orders_profits) - np.array(solver_profits)))
        print('variance of all_orders_profits - DNN_inference_profits',np.var(np.array(all_orders_profits) - np.array(DNN_inference_profits)))
        breakpoint()
    if tasks['price_quote_evaluation']:
        #just plot three arrays in the same graph and name it price quote investigation. For each array name them according to their variable name
        #and save the plot with name price_quote_investigation_combo2
        print(len(price_quote_all_markets),len(price_quote_solver_markets),len(price_quote_DNN_inference_markets))
        plt.figure(figsize=(12, 6))
        plt.plot(price_quote_all_markets, label="Price Quote of All Orders", marker='o')
        plt.plot(price_quote_solver_markets, label="Price Quote of Solver", marker='x')
        plt.plot(price_quote_DNN_inference_markets, label="Price Quote of DNN Inference", marker='s')
        plt.title("Price Quote Investigation", fontsize=14)
        plt.xlabel("Order Index", fontsize=12)
        plt.ylabel("Price Quote", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/common/home/hg343/Research/accelerate_combo_option/visualizations/combo_{single_stock}_price_quote_investigation_offset{offset}.png')

    #

def debug_market_comparison(test_all_orders_market, test_DNN_inference_market):
    """Debug market comparison and profit calculation"""
    
    # 1. Compare order compositions
    all_orders = test_all_orders_market.get_market_data_order_format()
    dnn_orders = test_DNN_inference_market.get_market_data_order_format()
    
    print("\nOrder Analysis:")
    print(f"All orders count: {len(all_orders)}")
    print(f"DNN orders count: {len(dnn_orders)}")
    
    # 2. Analyze payoff structure
    for strike in sorted(set(test_all_orders_market.get_strikes())):
        print(f"\nAt strike {strike}:")
        
        # For all orders market
        all_bid_sum = sum(max(row[0]*(strike-row[1]), 0) 
                         for _, row in all_orders.iterrows() if row[3] == 1)
        all_ask_sum = sum(max(row[0]*(strike-row[1]), 0) 
                         for _, row in all_orders.iterrows() if row[3] == 0)
        
        # For DNN market
        dnn_bid_sum = sum(max(row[0]*(strike-row[1]), 0) 
                         for _, row in dnn_orders.iterrows() if row[3] == 1)
        dnn_ask_sum = sum(max(row[0]*(strike-row[1]), 0) 
                         for _, row in dnn_orders.iterrows() if row[3] == 0)
        
        print(f"All orders - Bid sum: {all_bid_sum:.6f}, Ask sum: {all_ask_sum:.6f}")
        print(f"DNN orders - Bid sum: {dnn_bid_sum:.6f}, Ask sum: {dnn_ask_sum:.6f}")
    
    # 3. Check matched trades
    print("\nMatched Trades:")
    _, profit_all, matched_all, _ = Mechanism_solver_single(test_all_orders_market)
    _, profit_dnn, matched_dnn, _ = Mechanism_solver_single(test_DNN_inference_market)
    
    print(f"All orders profit: {profit_all:.6f}")
    print(f"DNN orders profit: {profit_dnn:.6f}")
    print(f"All orders matched indices: {matched_all}")
    print(f"DNN orders matched indices: {matched_dnn}")



