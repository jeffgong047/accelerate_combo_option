import pdb
import pickle
import argparse
import pandas as pd
import numpy as np
import random
import math
import os.path
from combinatorial.gen_synthetic_combo_options import gen_synthetic_combo_options
import itertools
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate stock option order books.')
    parser.add_argument('--num_stocks', type=int, default=2, help='Number of stocks to process (default: 2)')
    parser.add_argument('--num_orders', type=int, default=5000, help='Number of orders in the orderbook')
    parser.add_argument('--stock_combo', type=str, default=None, help='Comma-separated list of stock symbols to use (e.g. "AAPL,MSFT")')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for generating order books')
    parser.add_argument('--output_dir', type=str, default='data/orderbooks', help='Output directory for generated order books')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Available stocks
    selection = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 
                'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 
                'VZ', 'WBA', 'WMT', 'XOM']
    
    # Parse stock_combo argument if provided
    if args.stock_combo:
        stock_list = args.stock_combo.split(',')
        print(f"Using specified stock combination: {stock_list}")
    else:
        # Generate random combination
        possible_combinations = list(itertools.combinations(selection, args.num_stocks))
        stock_list = list(random.choice(possible_combinations))
        print(f"Using random stock combination: {stock_list}")
    
    # Seed for reproducibility
    random.seed(args.seed)
    
    # Join stock names for file naming
    combinations_string = '_'.join(stock_list)
    
    # Generate output filename
    filename = f'{args.output_dir}/STOCK_{args.num_stocks}_SEED_{args.seed}_book_{combinations_string}.npy'
    
    # Check if file already exists
    if os.path.isfile(filename):
        print(f"File {filename} already exists. Loading...")
        opt_book = np.load(filename)
        print(f"Loaded order book with shape {opt_book.shape}")
    else:
        print(f"Generating new synthetic options for {combinations_string}")
        opt_book, generated_stock_list = gen_synthetic_combo_options(
            NUM_ST=args.num_stocks, 
            NUM_ORDER=args.num_orders, 
            combinations=stock_list,
            SEED=args.seed
        )
        print(f"Generated order book with shape {opt_book.shape}")
        # Save the generated order book
        np.save(filename, opt_book)
        print(f"Saved order book to {filename}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())