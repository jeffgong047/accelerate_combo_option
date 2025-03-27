import pandas as pd
import numpy as np
import os
import sys 
from mechanism_solver import mechanism_solver_combo, mechanism_solver_single, synthetic_combo_match_mip
from copy import deepcopy
import pickle
import argparse 
import random
import signal
import re
from market_types import Market as MarketBase
from multiprocessing import Pool, TimeoutError
import traceback
from contextlib import contextmanager
import multiprocessing as mp

# Add this at the beginning of your script
def signal_handler(sig, frame):
    print('\nProgram interrupted by user. Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


'''
The script provides a class Market for financial_option_market that implements the following main functionalities:
1. epsilon_priceQuote: generate the frontier of options with epsilon price quote
2. epsilon_frontierGeneration: generate the frontier of options with epsilon price quote and constraints
'''


class Market(MarketBase):
    def __init__(self, opt_df: pd.DataFrame, mechanism_solver=None, input_format=None):
        '''
        The initialization should ensure the object could process all the functions in the class 
        opt_df: pandas dataframe of the market data columns: 
        columns: security coefficients, strike price, bid price, ask price, transaction type, liquidity
        input_format: the format of the market data, either 'option_series' or 'order format'   
        mechanism_solver: the mechanism solver to compute the profit of the given market. 
        If one wants to customized mechanism solver, one just need to ensure the input to the mechanism solver takes in orders in pandas dataframe format, and returns profit as first output.
        
        Strikes: market will prepare unique strikes for orders in the market and 0, infinity for constraints
        '''
        required_columns = ['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price', 'transaction_type']
        assert all(col in opt_df.columns for col in required_columns), "opt_df must contain all required columns"
        assert opt_df.index.is_unique, "opt_df index should be unique"
        self.opt_df = deepcopy(opt_df)
        if 'liquidity' not in opt_df.columns: # by default, we assume each order has only one unit of liquidity
            self.opt_df['liquidity'] = 1
        if input_format == 'option_series':
            self.opt_order = self.convert_maket_data_format(opt_df, format='order')
        else:
            self.opt_order = self.opt_df 
        self.strikes = list(set(self.opt_order.loc[:, 'Strike Price of the Option Times 1000']))

        self.mechanism_solver = mechanism_solver
    
    def remove_matched_orders(self, orders: pd.DataFrame=None, matched_orders_index: list=None):
        '''
        Remove the matched orders from the market
        '''
        if orders is None: 
            assert matched_orders_index is None, "matched_orders_index should be None if orders is None"
            orders = self.opt_order
            isMatch, profit, matched_order_index =  self.apply_mechanism(orders, offset = False, show_matched_orders=True)
        buy_book_index = matched_order_index['buy_book_index']
        sell_book_index = matched_order_index['sell_book_index']
        #sanity check: ensure buy and sell book index are not the same.
        shared_index = set(buy_book_index) & set(sell_book_index)
        assert len(shared_index) == 0, "buy and sell book index should not have any shared index"
        orders.drop(buy_book_index, inplace=True)
        orders.drop(sell_book_index, inplace=True)
        return orders

    def check_match(self, orders: pd.DataFrame=None, offset : bool = True):
        '''
        Check if the orders are matched
        '''
        if orders is None:
            orders = self.opt_order
        is_match, profit = self.apply_mechanism(orders, offset=offset)
        print('checking match', is_match, profit)
        return is_match, profit

    @staticmethod
    @contextmanager
    def pool_context(processes=None):
        pool = mp.Pool(processes=processes)
        try:
            yield pool
        finally:
            pool.terminate()
            pool.join()

    def run_mechanism_with_timeout(self, mechanism_solver, orders, offset=True, show_matched_orders=False):
        try:
            if mechanism_solver == mechanism_solver_combo:
                time, num_model_Constraints, profit, isMatch, matched_order_index = mechanism_solver(orders, offset=offset)
                if show_matched_orders:
                    return isMatch, profit, matched_order_index
                else:
                    return isMatch, profit
            elif mechanism_solver == mechanism_solver_single:
                market = Market(orders, input_format='order')
                return mechanism_solver(market, offset=offset)
            else:
                return mechanism_solver(orders)[0]
        except Exception as e:
            print(f"Process error: {str(e)}")
            traceback.print_exc()
            return None

    def apply_mechanism(self, orders: pd.DataFrame, offset: bool = True, show_matched_orders: bool = False, timeout=30):
        '''
        Apply the mechanism solver to the market data with timeout
        
        Args:
            orders: DataFrame containing order data
            offset: boolean for offset parameter
            show_matched_orders: whether to return matched order indices
            timeout: maximum time in seconds to wait for solver (default 5 minutes)
        '''
        assert not orders['liquidity'].isna().any(), "liquidity should not be nan"
        assert not orders['liquidity'].isnull().any(), "liquidity should not be null"
        
        if self.mechanism_solver is None:
            raise ValueError("Mechanism solver is not specified")

        with Market.pool_context(processes=1) as pool:
            try:
                async_result = pool.apply_async(
                    self.run_mechanism_with_timeout,
                    (self.mechanism_solver, orders, offset, show_matched_orders)
                )
                
                result = async_result.get(timeout=timeout)
                if result is None:
                    raise RuntimeError("Mechanism solver failed")
                return result
                
            except TimeoutError:
                print(f"Mechanism solver timed out after {timeout} seconds")
                raise
            except Exception as e:
                print(f"Error in mechanism solver: {str(e)}")
                raise

    def epsilon_priceQuote(self, option_to_quote : pd.DataFrame, orders_in_market : pd.DataFrame = None, offset : bool = True):
        '''
        quote price for option with epsilon amount,
        we only need to modify liquidity amount of the market orders
        '''

        if orders_in_market is None:
            orders_in_market = self.get_market_data_order_format()
        option_to_quote.loc[:, 'liquidity'] = 1
        orders_in_market.loc[:, 'liquidity'] = np.inf
        if option_to_quote.index[0] != 'quote':
            option_to_quote.index = ['quote']
        
        # FIX: Check if the specific index value 'quote' is in orders_in_market.index
        assert'quote' not in orders_in_market.index, "option_to_quote is already in the market"
        
        
        return self.priceQuote(option_to_quote, orders_in_market, offset)
            
    def priceQuote(self, option_to_quote : pd.DataFrame, orders_in_market : pd.DataFrame = None, liquidity: pd.Series = None, offset: bool = True):
        '''
        Generate the price of of given input order w.r.t orders in the market
        '''
        assert len(option_to_quote.index) == 1, "option_to_quote should have only one row"
        if orders_in_market is None:
            market_orders = self.get_market_data_order_format()
        else:
            market_orders = orders_in_market.copy()
        
        try:
            # Check for infinite liquidity properly
            is_match, profit = self.check_match(market_orders, offset=offset)
            if is_match and (market_orders['liquidity'] == np.inf).any():
                print(f"The market is matched, but contains infinite liquidity, cant get price quote")
                return None
            
            if option_to_quote.index[0] != 'quote':
                option_to_quote.index = ['quote']
            
            # Check if the specific index value is in market_orders.index
            assert option_to_quote.index[0] not in market_orders.index, "option_to_quote is already in the market"
            
            # Use iloc to access the first row regardless of index
            if option_to_quote.iloc[0]['transaction_type'] == 1:
                # quoting price for buy order, we want to quote price by adding a sell order with premium = 0 to the sell side of the market 
                new_sell_order = option_to_quote.copy()
                new_sell_order.iloc[0, new_sell_order.columns.get_loc('transaction_type')] = 0
                new_sell_order.iloc[0, new_sell_order.columns.get_loc('B/A_price')] = 0
                
                # Handle liquidity column
                if 'liquidity' in new_sell_order.columns:
                    if 'liquidity' in option_to_quote.columns:
                        new_sell_order.iloc[0, new_sell_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
                    else:
                        new_sell_order.iloc[0, new_sell_order.columns.get_loc('liquidity')] = 1
                else:
                    new_sell_order['liquidity'] = 1
                
                market_orders = pd.concat([market_orders, new_sell_order], ignore_index=False)
                try:
                    is_match, objVal = self.apply_mechanism(market_orders, offset)
                    if is_match:
                        bid_price = option_to_quote.iloc[0]['B/A_price']
                        quote_price = np.round(objVal, 2)
                        print(f'option_to_quote: {option_to_quote}')
                        print(f'quote price: {quote_price}')
                        
                        # Only return the price that matters for the frontier determination
                        return quote_price
                    else:
                        #need to debug this later but now forget about it 
                        return 0.01
                except TimeoutError:
                    print("Timeout in mechanism solver for buy order quote")
                    raise
                except Exception as e:
                    print(f"Error in mechanism solver for buy order quote: {e}")
                    return None
            elif option_to_quote.iloc[0]['transaction_type'] == 0:
                # quoting price for sell order, we want to quote price by adding a buy order with premium = max price to the buy side of the market 
                new_buy_order = option_to_quote.copy()
                new_buy_order.iloc[0, new_buy_order.columns.get_loc('transaction_type')] = 1
                new_buy_order.iloc[0, new_buy_order.columns.get_loc('B/A_price')] = 1e6  # Using 1e6 instead of sys.maxsize for numerical stability
                
                # Handle liquidity column
                if 'liquidity' in new_buy_order.columns:
                    if 'liquidity' in option_to_quote.columns:
                        new_buy_order.iloc[0, new_buy_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
                    else:
                        new_buy_order.iloc[0, new_buy_order.columns.get_loc('liquidity')] = 1
                else:
                    new_buy_order['liquidity'] = 1
                
                market_orders = pd.concat([market_orders, new_buy_order], ignore_index=False)
                try:
                    is_match, objVal = self.apply_mechanism(market_orders, offset)
                    if is_match:
                        ask_price = option_to_quote.iloc[0]['B/A_price']
                        quote_price = np.round(1e6 - objVal, 2)
                        return quote_price
                    else:
                        return 0.01             
                except TimeoutError:
                    print("Timeout in mechanism solver for sell order quote")
                    raise
                except Exception as e:
                    print(f"Error in mechanism solver for sell order quote: {e}")
                    return None
            else:
                raise ValueError("Invalid transaction type")
                
        except TimeoutError:
            raise  # Re-raise timeout to be handled by caller
        except Exception as e:
            print(f"Error in price quote: {e}")
            return None

    def frontierGeneration(self, orders : pd.DataFrame = None, epsilon : bool = False, offset : bool = True):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        if orders is None:
            orders = self.opt_order.copy()
        else:
            orders = orders.copy()
        
        frontier_labels = pd.Series(None, index=orders.index)
        quote_prices = pd.Series(None, index=orders.index)  # Create a Series to store quote prices
        
        try:
            for original_index, row_series in orders.iterrows():
                try:
                    print(f"Processing order {original_index} ({orders.index.get_loc(original_index)+1}/{len(orders)})")
                    
                    temp_orders = orders.copy()
                    order = row_series.to_frame().T
                    
                    original_price = row_series['B/A_price']
                    if original_price < 0:
                        print('Invalid original price')
                        frontier_labels[original_index] = None
                        continue
                        
                    order.index = ['quote']
                    order['liquidity'] = 1
                    
                    if epsilon:
                        temp_orders.loc[:, 'liquidity'] = np.inf
                    
                    temp_orders.drop(original_index, inplace=True)
                    
                    try:
                        quote_price = self.priceQuote(order, temp_orders, offset=offset)
                    except TimeoutError:
                        print(f"Timeout in price quote for order {original_index}")
                        breakpoint()
                        frontier_labels[original_index] = np.nan
                        continue
                    except Exception as e:
                        print(f"Error in price quote for order {original_index}: {e}")
                        breakpoint()
                        frontier_labels[original_index] = np.nan
                        continue
                    
                    if quote_price is None:
                        print(f"No valid quote price for order {original_index}")
                        breakpoint()
                        frontier_labels[original_index] = np.nan
                        continue
                    
                    # Store the quote price for this order
                    quote_prices[original_index] = quote_price
                        
                    print(f'quote_price: {quote_price}, original_price: {original_price}')
                    if original_price - quote_price < 1e-2 and original_price - quote_price > 0:
                        print('investigate')
                        breakpoint()
                    # Compare with the original price
                    if row_series['transaction_type'] == 1:  # Buy order
                        if original_price >= quote_price:
                            frontier_labels[original_index] = 1
                        else:
                            frontier_labels[original_index] = 0
                    else:  # Sell order
                        if original_price <= quote_price:
                            frontier_labels[original_index] = 1
                        else:
                            frontier_labels[original_index] = 0
                            
                except TimeoutError:
                    print(f"Timeout processing order {original_index}")
                    frontier_labels[original_index] = np.nan
                    breakpoint()
                except Exception as e:
                    print(f"Error processing order {original_index}: {e}")
                    frontier_labels[original_index] = np.nan
                    breakpoint()
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Returning partial results...")
        
        orders['belongs_to_frontier'] = frontier_labels
        print(orders)
        return orders, quote_prices

    def epsilon_frontierGeneration(self, orders : pd.DataFrame = None, offset : bool = True):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        if orders is None:
            orders = self.opt_order
        is_match, profit = self.check_match(orders)
        if is_match:
            print(f"The market is matched, cant get epsilon frontiers")
            return None
        else:
            labeled_orders, quote_prices = self.frontierGeneration(orders, epsilon = True, offset = offset)
            return labeled_orders, quote_prices

    def update_liquidity(self, liquidity : float):
        '''
        Update the liquidity of the orders in the market
        '''
        self.opt_order.loc[:, 'liquidity'] = liquidity

    def update_orders(self, orders : pd.DataFrame):
        '''
        Update the orders in the market
        '''
        self.opt_order = orders
        self.strikes = list(set(self.opt_order.loc[:, 'Strike Price of the Option Times 1000']))
        self.strikes.append(0)
        self.strikes.append(1e9)


    def drop_index(self, indices_to_drop):
        # Ensure indices_to_drop is a list
        if not isinstance(indices_to_drop, list):
            indices_to_drop = [indices_to_drop]

        # Drop indices from opt_df and opt_order
        self.opt_df.drop(indices_to_drop, inplace=True, errors='ignore')  # Use errors='ignore' to avoid KeyError
        self.opt_order.drop(indices_to_drop, inplace=True, errors='ignore')  # Use errors='ignore' to avoid KeyError

    def separate_buy_sell(self, orders=None):
        '''
        return [buy book, sell book]
        '''
        if orders is None:
            orders = self.opt_df
        return orders[orders['transaction_type'] == 1], orders[orders['transaction_type'] == 0]
    def get_strikes(self):
        return self.strikes 
    
    def get_market_data_raw(self):
        return self.opt_df.copy()
    
    def get_market_data_order_format(self):
        opt_order = self.opt_order.copy()
        # Get columns that match the pattern "option" followed by a number
        option_columns = [col for col in opt_order.columns if re.match(r'^option\d+$', col)]
        # Add option columns to the standard columns
        columns_to_return = option_columns + ['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price', 'transaction_type', 'liquidity'] 
        return opt_order.loc[:, columns_to_return].copy()
	
    def get_market_data_attrs(self):
        return self.opt_order.attrs.copy()
		
    def convert_market_data_format(self, opt_df, format='order'):
        '''
        This is particularly for raw data that each row contains both bid and ask price 
        order format: 
        columns: = ['C=Call, P=Put','Strike Price of the Option Times 1000','B/A_price','transaction_type', 'belongs_to_frontier', 'dominated_by', 'Unique ID for the Option Contract']
        '''

        if format == 'order':
            n_orders = len(opt_df)
            
            # Assert conditions
            if 'Expiration Date of the Option' in opt_df.columns and 'The Date of this Price' in opt_df.columns:
                assert len(opt_df['Expiration Date of the Option'].unique()) == 1, "All options must have the same expiration date"
                assert len(opt_df['The Date of this Price'].unique()) == 1, "All options must have the same price date"
            else:
                print("No expiration date or price date found in the input dataframe")
            # Initialize empty DataFrame with the correct columns
            opt = pd.DataFrame(columns=[
                'C=Call, P=Put', 
                'Strike Price of the Option Times 1000', 
                'B/A_price', 
                'transaction_type', 
                'belongs_to_frontier', 
                'dominated_by', 
                'Unique ID for the Option Contract'
            ], index=range(2 * n_orders))
            
            # Fill the DataFrame
            for i in range(n_orders):
                # First row: ask order
                opt.iloc[i] = [
                    1 if opt_df.iloc[i]['C=Call, P=Put'] == 'C' else -1,
                    opt_df.iloc[i]['Strike Price of the Option Times 1000']/1000,
                    opt_df.iloc[i]['Lowest  Closing Ask Across All Exchanges'],
                    0,  # transaction_type = 0 for ask
                    None,
                    [],
                    opt_df.iloc[i]['Unique ID for the Option Contract']
                ]
                
                # Second row: bid order
                opt.iloc[i + n_orders] = [
                    1 if opt_df.iloc[i]['C=Call, P=Put'] == 'C' else -1,
                    opt_df.iloc[i]['Strike Price of the Option Times 1000']/1000,
                    opt_df.iloc[i]['Highest Closing Bid Across All Exchanges'],
                    1,  # transaction_type = 1 for bid
                    None,
                    [],
                    opt_df.iloc[i]['Unique ID for the Option Contract']
                ]
            
            # Add attributes
            opt.attrs['expiration_date'] = opt_df['Expiration Date of the Option'].unique()[0]
            opt.attrs['The Date of this Price'] = opt_df['The Date of this Price'].unique()[0]
            
            return opt
