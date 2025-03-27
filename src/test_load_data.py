#!/usr/bin/env python3
"""
Simple test script to verify that data loading with the new offset_type parameter works correctly.
"""

import os
import sys
import glob
import pickle
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test loading frontier data with specific offset type")
    parser.add_argument('--data_dir', type=str, 
                        default='/common/home/hg343/Research/accelerate_combo_option/data/frontier_labels', 
                        help='Directory containing frontier label data')
    parser.add_argument('--offset_type', type=int, default=0, choices=[0, 1], 
                      help='Offset type to load data for: 0 = no offset, 1 = with offset')
    parser.add_argument('--noise_level', type=float, default=0.0625,
                      help='Noise level to filter files for (default: 0.0625)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    return args

def load_frontier_data(data_dir, seed=1, offset_type=0, noise_level=0.0625, combinations=None):
    """
    Load frontier data from the specified directory for given seed, offset type, and combinations.
    
    Args:
        data_dir: Directory containing frontier label data
        seed: Seed value to load
        offset_type: Offset type to load (0 or 1)
        noise_level: Noise level to filter for (default: 0.0625)
        combinations: List of stock combinations to load
        
    Returns:
        Dictionary with combination names as keys and lists of DataFrames as values
    """
    # Convert offset_type to corresponding directory name
    offset_dir = f'offset{offset_type}'
    noise_str = f'NOISE_{noise_level}'
    
    if combinations is None:
        combinations = ['BA_DIS', 'IBM_NKE', 'WMT_HD', 'GS_JPM', 'DIS_KO']
    
    data_dict = {}
    
    print(f"Looking for combinations: {combinations}")
    print(f"With offset_type: {offset_type} (dir: {offset_dir})")
    print(f"With noise level: {noise_level} (looking for '{noise_str}' in filenames)")
    print(f"Using seed: {seed}")
    
    for combo in combinations:
        combo_data = []
        
        # Path to the frontier data for this combination, seed and offset
        frontier_dir = os.path.join(data_dir, combo, f'seed{seed}', offset_dir)
        
        print(f"Checking directory: {frontier_dir}")
        
        if not os.path.exists(frontier_dir):
            print(f"Warning: Directory not found: {frontier_dir}")
            continue
            
        # Get all .pkl files that are not quote files
        frontier_files = glob.glob(os.path.join(frontier_dir, "*.pkl"))
        frontier_files = [f for f in frontier_files if not f.endswith('_quotes.pkl')]
        
        # Filter files to only include those with the specified noise level
        filtered_files = [f for f in frontier_files if noise_str in f]
        
        print(f"Found {len(filtered_files)} out of {len(frontier_files)} files with noise level {noise_level} in {frontier_dir}")
        
        for file_path in filtered_files[:2]:  # Just load two files for testing
            try:
                print(f"Loading file: {os.path.basename(file_path)}")
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Convert to DataFrame if it's a dictionary
                if isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    df = data
                    
                combo_data.append(df)
                
                # Print some statistics
                num_asks = len(df[df['transaction_type'] == 0])
                num_bids = len(df[df['transaction_type'] == 1])
                num_frontier_asks = len(df[(df['transaction_type'] == 0) & (df['belongs_to_frontier'] == 1)])
                num_frontier_bids = len(df[(df['transaction_type'] == 1) & (df['belongs_to_frontier'] == 1)])
                
                print(f"  - File contains {len(df)} orders")
                print(f"  - Asks: {num_asks}, Frontier asks: {num_frontier_asks}")
                print(f"  - Bids: {num_bids}, Frontier bids: {num_frontier_bids}")
                print(f"  - Frontier ratio: {(num_frontier_asks + num_frontier_bids) / len(df):.4f}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if combo_data:
            data_dict[combo] = combo_data
            print(f"Successfully loaded {len(combo_data)} datasets for {combo}")
    
    return data_dict

def main():
    args = parse_arguments()
    print(f"Testing data loading with offset_type={args.offset_type} and noise_level={args.noise_level}")
    
    # Load data
    data_dict = load_frontier_data(args.data_dir, seed=args.seed, offset_type=args.offset_type, noise_level=args.noise_level)
    
    # Print summary
    print("\nSummary:")
    print("-" * 40)
    if not data_dict:
        print(f"No data found for offset_type={args.offset_type} and noise_level={args.noise_level}")
        return
    
    for combo, datasets in data_dict.items():
        print(f"Combo: {combo}, Datasets: {len(datasets)}")
    
    print("\nTest complete.")

if __name__ == "__main__":
    main() 