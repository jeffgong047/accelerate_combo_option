import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
# from match_prediction import BiAttentionClassifier_single as BiAttentionClassifier
from match_prediction import train_model, validate_model, test_model, BiAttentionClassifier, test_model_online, finetune_policy_head, evaluate_policy_head
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
import numpy as np
import pickle
from utils import collate_fn, profit_with_penalty_reward, profit_minus_liability_reward, get_reward_function
import os 
from tqdm import tqdm 
import itertools
from match_prediction.reinforcement_learning import train_dqn_for_market_matching, dqn_evaluate_market
import time
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=6, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test set size as a percentage of the entire data')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations to train the model')
    parser.add_argument('--market_size', type=int, default=50, help='Market size')
    parser.add_argument('--offset_type', type=int, default=0, choices=[0, 1], 
                      help='Offset type to load data for: 0 = no offset, 1 = with offset')
    parser.add_argument('--tasks', type=str, nargs='+', default=['next_order_frontier_prediction'], help='List of tasks to perform (space-separated)')# 'batch_profit_evaluation', 'next_order_frontier_prediction','price_quote_evaluation'
    parser.add_argument('--test_dqn', action='store_true', help='Whether to test DQN approach')
    parser.add_argument('--dqn_episodes', type=int, default=20, help='Number of episodes for DQN training')
    parser.add_argument('--reward_type', type=str, default='profit_with_penalty', 
                      choices=['profit_with_penalty', 'profit_minus_liability'], 
                      help='Type of reward function to use')
    parser.add_argument('--penalty_weight', type=float, default=20, help='Weight for selection penalty')
    parser.add_argument('--liability_weight', type=float, default=0.1, help='Weight for liability penalty')
    parser.add_argument('--data_dir', type=str, default='/common/home/hg343/Research/accelerate_combo_option/data/frontier_labels', 
                        help='Directory containing frontier label data')
    parser.add_argument('--model_dir', type=str, default='/common/home/hg343/Research/accelerate_combo_option/models', 
                        help='Directory to save trained models')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
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

def load_frontier_data(data_dir, seed=1, offset_type=0, combinations=None, noise_level=0.0015625):
    """
    Load frontier data from the specified directory for given seed, offset type, and combinations.
    
    Args:
        data_dir: Directory containing frontier label data
        seed: Seed value to load
        offset_type: Offset type to load (0 or 1)
        combinations: List of stock combinations to load
        noise_level: Noise level to filter for (default: 0.0625 which is 2^-4)
        
    Returns:
        Dictionary with combination names as keys and lists of DataFrames as values
    """
    noise_level=0.015625
    # Convert offset_type to corresponding directory name
    offset_dir = f'offset{offset_type}'
    noise_str = f'NOISE_{noise_level}'
    
    if combinations is None:
        combinations = ['BA_DIS', 'IBM_NKE', 'WMT_HD', 'GS_JPM', 'DIS_KO']
    
    data_dict = {}
    
    print(f"Loading only files with noise level: {noise_level} (looking for '{noise_str}' in filenames)")
    
    for combo in combinations:
        combo_data = []
        
        # Path to the frontier data for this combination, seed and offset
        frontier_dir = os.path.join(data_dir, combo, f'seed{seed}', offset_dir)
        if not os.path.exists(frontier_dir):
            print(f"Warning: Directory not found: {frontier_dir}")
            continue
            
        # Get all .pkl files that are not quote files
        frontier_files = glob.glob(os.path.join(frontier_dir, "*.pkl"))
        frontier_files = [f for f in frontier_files if not f.endswith('_quotes.pkl')]
        
        # Filter files to only include those with the specified noise level
        filtered_files = [f for f in frontier_files if noise_str in f]
        
        print(f"Found {len(filtered_files)} out of {len(frontier_files)} files with noise level {noise_level} in {frontier_dir}")
        
        for file_path in filtered_files:
            print(file_path)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Convert to DataFrame if it's a dictionary
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                
                combo_data.append(df)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if combo_data:
            data_dict[combo] = combo_data
            print(f"Loaded {len(combo_data)} datasets for {combo}")
    
    return data_dict

def main():
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set noise level to use
    noise_level = 0.0625 # 2^-4
    
    # Default model path - include offset_type and noise level in the filename
    model_path = os.path.join(args.model_dir, 
                             f'frontier_model_offset{args.offset_type}_noise{noise_level}_layers{args.num_layers}_hidden{args.hidden_size}.pth')
    
    # Load data from the frontier_labels directory for the specified offset type and noise level only
    data_dict = load_frontier_data(args.data_dir, seed=args.seed, offset_type=args.offset_type, noise_level=noise_level)
    
    if not data_dict:
        print(f"Error: No frontier data found for offset type {args.offset_type} and noise level {noise_level}. Please check the data directory.")
        return
    
    # Print statistics about the loaded data
    print("\n"+"-"*60)
    print(f"FRONTIER STATISTICS FOR OFFSET TYPE {args.offset_type} AND NOISE LEVEL {noise_level}")
    print("-"*60)
    
    # Calculate overall statistics for each combination
    total_stats = {}
    
    for combo_name, combo_data in data_dict.items():
        total_orders = 0
        total_frontier_orders = 0
        
        for df in combo_data:
            # Count total orders in this market
            market_total = len(df)
            # Count orders that belong to frontier
            market_frontier = len(df[df['belongs_to_frontier'] == 1])
            
            total_orders += market_total
            total_frontier_orders += market_frontier
        
        # Calculate frontier percentage for this combination
        frontier_percentage = (total_frontier_orders / total_orders * 100) if total_orders > 0 else 0
        
        # Store statistics for later use
        total_stats[combo_name] = {
            'total_orders': total_orders,
            'frontier_orders': total_frontier_orders,
            'frontier_percentage': frontier_percentage
        }
        
        # Print statistics for this combination
        print(f"Combination: {combo_name}")
        print(f"  - Total orders: {total_orders}")
        print(f"  - Orders on frontier: {total_frontier_orders}")
        print(f"  - Frontier percentage: {frontier_percentage:.2f}%")
        print()
    
    # Calculate combined statistics for all combinations
    all_orders = sum(stats['total_orders'] for stats in total_stats.values())
    all_frontier = sum(stats['frontier_orders'] for stats in total_stats.values())
    overall_percentage = (all_frontier / all_orders * 100) if all_orders > 0 else 0
    
    print("-"*60)
    print(f"COMBINED STATISTICS ACROSS ALL COMBINATIONS:")
    print(f"  - Total orders across all combinations: {all_orders}")
    print(f"  - Total orders on frontier: {all_frontier}")
    print(f"  - Overall frontier percentage: {overall_percentage:.2f}%")
    print("-"*60 + "\n")
    
    # Merge all stock combination data together for training
    print("Merging all stock combination data together for training...")
    all_combo_data = []
    for combo_name, combo_data in data_dict.items():
        all_combo_data.extend(combo_data)
        print(f"Added {len(combo_data)} markets from {combo_name}")
    
    print(f"Total markets after merging: {len(all_combo_data)}")
    breakpoint()
    # Add noisy versions of markets that create matches for training
    noisy_markets = []
    for index, market in enumerate(all_combo_data):
        if index > 1000:  # Limit to 1000 markets for noise addition
            break
        noisy_market = add_noise_to_create_match(market)
        noisy_markets.append(noisy_market)
    
    # Create separate noisy markets for evaluation
    eval_noisy_markets = []
    # Use a different slice of data for evaluation
    for index, market in enumerate(all_combo_data[1001:1201]):  # Use 200 markets for evaluation
        eval_noisy_market = add_noise_to_create_match(market)
        eval_noisy_markets.append(eval_noisy_market)
    
    print(f"Added {len(noisy_markets)} matching markets for training through noise addition")
    print(f"Added {len(eval_noisy_markets)} matching markets for evaluation through noise addition")
    
    # Define features order
    features_order = ['option1', 'option2', 'C=Call, P=Put',
                    'Strike Price of the Option Times 1000',
                    'B/A_price', 'transaction_type']
    
    # Prepare data for training
    all_combo_features = [x[features_order].to_numpy(dtype=np.float32) for x in all_combo_data]
    all_combo_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in all_combo_data]
    
    # Train-test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_combo_features, all_combo_labels, test_size=args.test_split, random_state=args.seed
    )
    
    # Calculate percentage of frontier labels
    y_all_temp = np.concatenate(all_combo_labels)
    percentage_labels_1_all = sum(y_all_temp) / len(y_all_temp) * 100
    print(f"Percentage of labels = 1 across all combinations with offset {args.offset_type} and noise {noise_level}: {percentage_labels_1_all:.2f}%")
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(1 - args.train_split), random_state=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = list(zip(X_test, y_test))
    test_loader_online = DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size, 
                                  num_layers=args.num_layers, num_classes=2)
    
    # Load existing model if available
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    frontier_loss_fn = nn.CrossEntropyLoss().float()
    
    # Train the model if not already trained
    if not os.path.exists(model_path):
        print("Training new model...")
        for i in tqdm(range(args.iterations), desc='Training'):
            train_model(model, train_loader, optimizer, frontier_loss_fn, epochs=args.epochs)
            validate_model(model, val_loader, frontier_loss_fn)
    
    # Test model performance
    print("Performance on test data:")
    test_model(model, test_loader)
    
    # Save the model if newly trained
    if not os.path.exists(model_path):
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
    
    # Prepare data for DQN training from noisy markets
    if noisy_markets:
        noisy_features = [x[features_order].to_numpy(dtype=np.float32) for x in noisy_markets]
        noisy_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in noisy_markets]
        
        # Create a DataLoader with the noisy markets for training
        noisy_loader = DataLoader(
            list(zip(noisy_features, noisy_labels)), 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Prepare evaluation data
        eval_noisy_features = [x[features_order].to_numpy(dtype=np.float32) for x in eval_noisy_markets]
        eval_noisy_labels = [x['belongs_to_frontier'].to_numpy(dtype=np.float32) for x in eval_noisy_markets]
        eval_noisy_data = list(zip(eval_noisy_features, eval_noisy_labels))
        
        # Test DQN if requested
        if args.test_dqn:
            print(f'Testing DQN approach with reward_type: {args.reward_type}, penalty_weight: {args.penalty_weight}, liability_weight: {args.liability_weight}')
            
            dqn_metrics, baseline_metrics, dqn_agent = test_dqn_approach(
                model=model,
                train_loader=noisy_loader,
                test_loader=eval_noisy_data,
                reward_fn=synthetic_combo_match_mip,
                features_order=features_order,
                **vars(args)
            )
            
            # Save the trained DQN agent - include offset_type and noise level in filename
            dqn_model_path = os.path.join(args.model_dir, f'frontier_dqn_offset{args.offset_type}_noise{noise_level}_{args.reward_type}_seed{args.seed}.pt')
            torch.save({
                'dqn_state_dict': dqn_agent.q_network.state_dict(),
                'target_state_dict': dqn_agent.target_network.state_dict(),
                'metrics': dqn_metrics,
                'reward_type': args.reward_type
            }, dqn_model_path)
            print(f"DQN model saved to {dqn_model_path}")

def test_dqn_approach(model, train_loader, test_loader, reward_fn, features_order, **args):
    """
    Test the DQN approach for market order matching and compare with the original approach
    
    Args:
        model: Pretrained model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        reward_fn: Function for matching (e.g., synthetic_combo_match_mip)
        features_order: Feature names for DataFrame conversion
        **args: Additional arguments
    """
    from match_prediction.reinforcement_learning import train_dqn_for_market_matching, dqn_evaluate_market
    import time
    
    print("="*80)
    print("Testing DQN Approach for Market Order Matching")
    print("="*80)
    
    # Store the original model weights to restore later
    original_weights = {name: param.clone() for name, param in model.state_dict().items()}

    # Get reward type from args
    reward_type = args.get('reward_type', 'profit_with_penalty')
    print(f"Using reward function: {reward_type}")
    
    # Additional reward function parameters
    reward_kwargs = {}
    if reward_type == 'profit_with_penalty':
        reward_kwargs['penalty_weight'] = args.get('penalty_weight', 0.1)
    elif reward_type == 'profit_minus_liability':
        reward_kwargs['liability_weight'] = args.get('liability_weight', 0.1)
    
    # Initialize DQN training parameters
    dqn_params = {
        'num_episodes': args.get('dqn_episodes', 20), 
        'batch_size': args.get('batch_size', 32),
        'target_update': 10,
        'hidden_size': args.get('hidden_size', 64),
        'lr': args.get('learning_rate', 1e-4) * 0.1,  # Lower learning rate for DQN
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.95,
        'reward_type': reward_type,
        **reward_kwargs
    }
    
    # Start timing
    start_time = time.time()
    
    # Train DQN agent
    print("Training DQN agent...")
    agent = train_dqn_for_market_matching(
        model=model,
        train_loader=train_loader,
        reward_fn=reward_fn,
        features=features_order,
        **dqn_params
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"DQN training completed in {train_time:.2f} seconds")
    
    # Start timing for evaluation
    start_time = time.time()
    
    # Evaluate DQN agent
    print("\nEvaluating DQN agent...")
    dqn_metrics = dqn_evaluate_market(
        model=model,
        agent=agent,
        test_loader=test_loader,
        reward_fn=reward_fn,
        features=features_order
    )
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    print(f"DQN evaluation completed in {eval_time:.2f} seconds")
    
    # Restore original model weights
    model.load_state_dict(original_weights)
    
    # Start timing for baseline approach
    start_time = time.time()
    
    # Test with original approach
    print("\nEvaluating baseline approach...")
    from match_prediction import evaluate_policy_head
    baseline_metrics = evaluate_policy_head(
        model=model,
        test_loader=test_loader,
        reward_fn=reward_fn,
        features_order=features_order,
        **args
    )
    
    # Calculate baseline evaluation time
    baseline_time = time.time() - start_time
    print(f"Baseline evaluation completed in {baseline_time:.2f} seconds")
    
    # Compare metrics
    print("\n" + "="*80)
    print("Comparison of DQN vs Baseline Approach")
    print("="*80)
    print(f"{'Metric':<20} {'DQN':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-"*80)
    
    # Convert baseline metrics to match DQN metrics format
    if isinstance(baseline_metrics, tuple):
        baseline_avg_profit = baseline_metrics[0]
        baseline_match_rate = baseline_metrics[1] 
        baseline_selection_ratio = baseline_metrics[2]
    else:
        baseline_avg_profit = baseline_metrics.get('avg_profit', 0)
        baseline_match_rate = baseline_metrics.get('match_rate', 0)
        baseline_selection_ratio = baseline_metrics.get('selection_ratio', 0)
    
    # Print comparison metrics
    profit_improvement = (dqn_metrics['avg_profit'] / baseline_avg_profit - 1) * 100 if baseline_avg_profit > 0 else float('inf')
    print(f"{'Avg Profit':<20} {dqn_metrics['avg_profit']:<15.4f} {baseline_avg_profit:<15.4f} {profit_improvement:<15.2f}%")
    
    match_improvement = (dqn_metrics['match_rate'] / baseline_match_rate - 1) * 100 if baseline_match_rate > 0 else float('inf')
    print(f"{'Match Rate':<20} {dqn_metrics['match_rate']:<15.4f} {baseline_match_rate:<15.4f} {match_improvement:<15.2f}%")
    
    selection_diff = (dqn_metrics['selection_ratio'] - baseline_selection_ratio) * 100
    print(f"{'Selection Ratio':<20} {dqn_metrics['selection_ratio']:<15.4f} {baseline_selection_ratio:<15.4f} {selection_diff:<15.2f}%")
    
    time_diff = (baseline_time / eval_time - 1) * 100 if eval_time > 0 else float('inf')
    print(f"{'Eval Time (s)':<20} {eval_time:<15.2f} {baseline_time:<15.2f} {time_diff:<15.2f}%")
    
    return dqn_metrics, baseline_metrics, agent

if __name__ == '__main__':
    main() 