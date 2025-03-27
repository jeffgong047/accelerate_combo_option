import pdb
import pickle
import argparse
import pandas as pd
import numpy as np
import random
import math
import os.path
from combinatorial.gen_synthetic_combo_options import gen_synthetic_combo_options
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
from mechanism_solver import mechanism_solver_combo
from gurobipy import *
import timeit
from copy import deepcopy
from tqdm import tqdm
# Run in a separate process with timeout
from multiprocessing import Process, Queue, Pool
import multiprocessing as mp 
import queue
import traceback
from contextlib import contextmanager
import signal
import sys
from multiprocessing import Pool, TimeoutError 
import itertools
import os 
from market import Market
from combo_stock_frontier_data_preprocessor import synthetic_combo_frontier_generation

# Add this function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process stock options.')
    parser.add_argument('--num_stocks', type=int, default=2, help='Number of stocks to process (default: 3)')
    parser.add_argument('--market_size', type=int, default=50, help='Number of orders in the market')
    parser.add_argument('--offset', type=bool, default=False, help='Whether to allow offset for liability in the optimization')
    parser.add_argument('--wandb_project', type=str, default='expediating_comb_financial_market_matching', help='Wandb project name (not used, kept for compatibility)')
    parser.add_argument('--num_orders', type=int, default=5000, help='number of orders in the orderbook')
    parser.add_argument('--noise', type=float, default=2**-4, help='noise level in the orderbook')
    parser.add_argument('--stock_combo', type=str, default=None, help='Comma-separated list of stock symbols to use (e.g. "AAPL,MSFT")')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for generating order books')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for generated order books')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with breakpoints')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

# # Parse arguments
args = parse_arguments()

def signal_handler(signum, frame):
    print("Ctrl+C received. Terminating processes...")
    if 'pool' in globals():
        pool.terminate()
        pool.join()
    sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def add_noise_orderbook(opt_book, NOISE=0.01):
    SEED = 1
    random.seed(SEED)
    # coeff up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
    opt_buy_book, opt_sell_book = opt_book[opt_book[:, -2]==1], opt_book[opt_book[:, -2]==0]
    num_buy, num_sell = len(opt_buy_book), len(opt_sell_book)
    # add noise
    buy_noise = [random.random()*NOISE+1 for i in range(num_buy)]
    opt_buy_book[:, -1] = np.round(buy_noise * opt_buy_book[:, -1], 2)
    sell_noise = [1-random.random()*NOISE for i in range(num_sell)]
    opt_sell_book[:, -1] = np.round(sell_noise * opt_sell_book[:, -1], 2)
    print('There are {} buy orders and {} sell orders'.format(num_buy, num_sell))
    return opt_buy_book, opt_sell_book

@contextmanager
def pool_context(processes=None):
    pool = mp.Pool(processes=processes)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()

def log(message, verbose=False):
    """Helper function to log messages based on verbosity"""
    if verbose or args.verbose:
        print(f"[INFO] {message}")

if __name__ == '__main__':
    NUM_STOCK = args.num_stocks
    MARKET_SIZE = args.market_size
    NOISE = args.noise
    BOOK_SIZE = args.market_size
    SEED = args.seed
    
    print(f"Starting processing with the following parameters:")
    print(f"  - Number of stocks: {NUM_STOCK}")
    print(f"  - Market size: {MARKET_SIZE}")
    print(f"  - Noise level: {NOISE}")
    print(f"  - Seed: {SEED}")
    print(f"  - Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"  - Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    
    tasks = {}
    directory_path = args.output_dir if args.output_dir else f'/common/home/hg343/Research/accelerate_combo_option/data/combo_2_test'
    
    try:
        selection = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM']
        
        # Debug breakpoint - only if debug mode is enabled
        if args.debug:
            print("Debug mode enabled, entering initial breakpoint")
            breakpoint()
            
        # Use stock_combo parameter if provided
        if args.stock_combo:
            stock_list = args.stock_combo.split(',')
            print('Stock list being processed:', stock_list)
        else:
            raise ValueError('No stock combination provided')
            
        combinations_string = '_'.join(stock_list)
        print(f"Combination string: {combinations_string}")
        
        with pool_context(processes=min(20, os.cpu_count())) as pool:
            # Only process seed 1 instead of looping through multiple seeds
            try:
                # Check for the orderbook at the expected locations
                filename = f'combinatorial/book/STOCK_{NUM_STOCK}_SEED_{SEED}_book_{combinations_string}.npy'
                input_file = f'/common/home/hg343/Research/accelerate_combo_option/data/generated_orderbooks/combo2/STOCK_{NUM_STOCK}_SEED_{SEED}_book_{combinations_string}.npy'
                
                print(f"Looking for orderbook file in: \n  1. {input_file}\n  2. {filename}")
                
                # First check in the data directory
                if os.path.isfile(input_file):
                    print(f"Found and loading orderbook from: {input_file}")
                    opt_book = np.load(input_file)
                    print(f"Successfully loaded orderbook with shape: {opt_book.shape}")
                # Then check in the local directory
                elif os.path.isfile(filename):
                    print(f"Found and loading orderbook from: {filename}")
                    opt_book = np.load(filename)
                    print(f"Successfully loaded orderbook with shape: {opt_book.shape}")
                else:
                    # Instead of generating, raise an error
                    raise FileNotFoundError(f"Orderbook file not found for combination '{combinations_string}' with seed {SEED}. "
                                          f"Looked in paths: {input_file} and {filename}")
                    
                num_books = len(opt_book)//MARKET_SIZE
                print(f"Found {num_books} potential markets in the orderbook")
                
                market_count = 0
                success_count = 0
                
                for market_index in tqdm(range(1, num_books), desc=f'Generating frontier for markets'):
                    try:
                        market_count += 1
                        stock_name = '_'.join(stock_list)
                        log(f"Processing market {market_index} (total processed: {market_count})")
                        
                        opt_book_1 = opt_book[market_index*MARKET_SIZE:(market_index+1)*MARKET_SIZE]
                        log(f"Extracted market segment with {len(opt_book_1)} orders")
                        
                        opt_buy_book, opt_sell_book = add_noise_orderbook(opt_book_1, NOISE)
                        column_names = ['option1', 'option2', 'C=Call, P=Put', 'Strike Price of the Option Times 1000', 'transaction_type', 'B/A_price']
                        opt_orders_df = pd.DataFrame(np.concatenate([opt_buy_book, opt_sell_book], axis=0), columns=column_names)
                        
                        print(f'#####Generating market {market_index} with size {BOOK_SIZE} and noise {NOISE}#####')
                        log(f"Created orders DataFrame with shape: {opt_orders_df.shape}")
                        
                        # Include offset in filename
                        offset_str = 'with_offset' if args.offset else 'no_offset'
                        filename = f'frontier_market_{market_index}_book_size_{BOOK_SIZE}_{stock_name}_NOISE_{NOISE}_{offset_str}'
                        
                        log(f"Creating Market object...")
                        market = Market(opt_orders_df, mechanism_solver=mechanism_solver_combo)
                        log(f"Checking for matches...")
                        is_match, profit = market.check_match()
                        if is_match:
                            log(f"Found match with profit {profit}, removing matched orders")
                            market.remove_matched_orders()
                        
                        # Only enter debug mode if --debug flag is set
                        if args.debug:
                            print("Debug mode enabled, entering breakpoint before frontier generation")
                            breakpoint()
                            
                        print(f"Starting frontier generation with offset={args.offset}")
                        frontier_option_label_epsilon, quote_price_epsilon = market.epsilon_frontierGeneration(offset=args.offset)
                        
                        # Handle the case where frontier_option_label_epsilon is None
                        if frontier_option_label_epsilon is None:
                            print(f"Iteration {market_index} returned None frontier_option_label_epsilon")
                            continue
                        
                        # Check for DataFrame (some versions might return DataFrame)
                        if isinstance(frontier_option_label_epsilon, pd.DataFrame):
                            log(f"frontier_option_label_epsilon data shape: {frontier_option_label_epsilon.shape}")
                            if quote_price_epsilon is not None and isinstance(quote_price_epsilon, pd.DataFrame):
                                log(f"quote_price_epsilon data shape: {quote_price_epsilon.shape}")
                        else:
                            # Handle non-dataframe case (dictionary or other format)
                            log(f"frontier_option_label_epsilon type: {type(frontier_option_label_epsilon)}")
                            log(f"quote_price_epsilon type: {type(quote_price_epsilon) if quote_price_epsilon is not None else 'None'}")
                        
                        # Check if returned data is valid before saving
                        valid_frontier = False
                        if isinstance(frontier_option_label_epsilon, pd.DataFrame):
                            if 'belongs_to_frontier' in frontier_option_label_epsilon.columns and not frontier_option_label_epsilon['belongs_to_frontier'].isna().any():
                                valid_frontier = True
                        elif isinstance(frontier_option_label_epsilon, dict):
                            if 'belongs_to_frontier' in frontier_option_label_epsilon and not np.isnan(frontier_option_label_epsilon['belongs_to_frontier']).any():
                                valid_frontier = True
                        
                        if valid_frontier:
                            success_count += 1
                            print(f"Successfully completed iteration {market_index} ({success_count}/{market_count} successful)")
                            
                            if not os.path.exists(directory_path):
                                os.makedirs(directory_path)
                                
                            save_path = os.path.join(directory_path, filename + '.pkl')
                            log(f"Saving frontier data to: {save_path}")
                            
                            with open(save_path, 'wb') as f:
                                pickle.dump(frontier_option_label_epsilon, f)
                                
                            # Save quote prices separately
                            quotes_path = os.path.join(directory_path, filename + '_quotes.pkl')
                            log(f"Saving quote prices to: {quotes_path}")
                            with open(quotes_path, 'wb') as f:
                                pickle.dump(quote_price_epsilon, f)
                                
                            print(f"Successfully saved data for market {market_index}")
                        else:
                            print(f"Iteration {market_index} returned invalid or NaN-containing data")
                            
                    except TimeoutError:
                        print(f"Market {market_index} processing timed out")
                        continue
                    except Exception as e:
                        print(f"Error processing market {market_index}: {str(e)}")
                        traceback.print_exc()
                        continue
                
                print(f"Processing complete: {success_count} out of {market_count} markets processed successfully")
                
            except Exception as e:
                print(f"Error processing combination {combinations_string}: {str(e)}")
                traceback.print_exc()
                raise e
                    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        raise e
