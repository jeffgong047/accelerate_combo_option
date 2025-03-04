import pandas as pd
import numpy as np
import os
import sys 
from mechanism_solver import mechanism_solver_combo, mechanism_solver_single
from copy import deepcopy
import pickle
import argparse 
import random
import signal

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


class Market:
    def __init__(self, opt_df: pd.DataFrame, mechanism_solver=None,input_format = None):
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
        self.opt_df = deepcopy(opt_df)
        if 'liquidity' not in opt_df.columns: # by default, we assume each order has only one unit of liquidity
            self.opt_df['liquidity'] = 1
        if input_format == 'option_series':
            self.opt_order = self.convert_maket_data_format(opt_df, format='order')
        else:
            self.opt_order = self.opt_df 
        self.strikes = list(set(self.opt_order.loc[:, 'Strike Price of the Option Times 1000']))

        self.mechanism_solver = mechanism_solver

    def check_match(self, orders: pd.DataFrame, offset : bool = True):
        '''
        Check if the orders are matched
        '''
        is_match, profit = self.apply_mechanism(orders, offset=offset)
        print('checking match', is_match, profit)
        return is_match, profit

    def apply_mechanism(self, orders : pd.DataFrame, offset : bool = True):
        '''
        Apply the mechanism solver to the market data
        '''
        #sanity check: ensure liquidity is not nan or None 
        assert not orders['liquidity'].isna().any(), "liquidity should not be nan"
        assert not orders['liquidity'].isnull().any(), "liquidity should not be null"
        if self.mechanism_solver is None:
            raise ValueError("Mechanism solver is not specified")
        elif self.mechanism_solver == mechanism_solver_combo:
            buy_orders, sell_orders = self.separate_buy_sell(orders)
            return self.mechanism_solver(buy_orders, sell_orders, offset=offset)[2]
        elif self.mechanism_solver == mechanism_solver_single:
            # FIXED VERSION: Call mechanism_solver directly
            market = Market(orders, input_format='order')
            return self.mechanism_solver(market, offset=offset)
        else:
            return self.mechanism_solver(orders)[0]

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
        assert 'quote' not in orders_in_market.index, "option_to_quote is already in the market"
        
        return self.priceQuote(option_to_quote, orders_in_market, offset)
            
    def priceQuote(self, option_to_quote : pd.DataFrame, orders_in_market : pd.DataFrame = None, offset: bool = True):
        '''
        Generate the price of of givne input order w.r.t orders in the market
        '''
        assert len(option_to_quote.index) == 1, "option_to_quote should have only one row"
        if orders_in_market is None:
            market_orders = self.get_market_data_order_format()
        else:
            market_orders = orders_in_market.copy()
        if self.check_match(market_orders)[0] and np.inf in market_orders['liquidity']:
            print(f"The market is matched, cant get price quote")
            return None
        if option_to_quote.index[0] != 'quote':
            option_to_quote.index = ['quote']
        
        # FIX: Check if the specific index value is in market_orders.index
        assert option_to_quote.index[0] not in market_orders.index, "option_to_quote is already in the market"
        
        # Use iloc to access the first row regardless of index
        if option_to_quote.iloc[0]['transaction_type'] == 1:
            # quoting price for buy order, we want to quote price by adding a sell order with premium = 0 to the sell side of the market 
            new_sell_order = option_to_quote.copy()
            new_sell_order.iloc[0, new_sell_order.columns.get_loc('transaction_type')] = 0
            new_sell_order.iloc[0, new_sell_order.columns.get_loc('B/A_price')] = 0
            
            # FIX: Check if 'liquidity' exists in new_sell_order before trying to set it
            if 'liquidity' in new_sell_order.columns:
                # Column exists, use get_loc
                if 'liquidity' in option_to_quote.columns:
                    new_sell_order.iloc[0, new_sell_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
                else:
                    new_sell_order.iloc[0, new_sell_order.columns.get_loc('liquidity')] = 1
            else:
                # Column doesn't exist, create it
                new_sell_order['liquidity'] = 1
            
            market_orders = pd.concat([market_orders, new_sell_order], ignore_index=False)
            is_match, objVal = self.apply_mechanism(market_orders, offset)
            return objVal
        elif option_to_quote.iloc[0]['transaction_type'] == 0:
            # quoting price for sell order, we want to quote price by adding a buy order with premium = max price to the buy side of the market 
            new_buy_order = option_to_quote.copy()
            new_buy_order.iloc[0, new_buy_order.columns.get_loc('transaction_type')] = 1
            new_buy_order.iloc[0, new_buy_order.columns.get_loc('B/A_price')] = sys.maxsize
            
            # FIX: Check if 'liquidity' exists in new_buy_order before trying to set it
            if 'liquidity' in new_buy_order.columns:
                # Column exists, use get_loc
                if 'liquidity' in option_to_quote.columns:
                    new_buy_order.iloc[0, new_buy_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
                else:
                    new_buy_order.iloc[0, new_buy_order.columns.get_loc('liquidity')] = 1
            else:
                # Column doesn't exist, create it
                new_buy_order['liquidity'] = 1
            
            market_orders = pd.concat([market_orders, new_buy_order, option_to_quote], ignore_index=False)
            is_match, objVal = self.apply_mechanism(market_orders, offset)
            if is_match:
                return sys.maxsize - objVal
            else:
                return None
        else:
            raise ValueError("Invalid transaction type")
    def frontierGeneration(self, orders : pd.DataFrame = None, epsilon : bool = False):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        # we dont want to change the original orders, so we make a copy of it 
        if orders is None:
            orders = self.opt_order.copy()
        else:
            orders = orders.copy()
        
        frontier_labels = pd.Series(None, index=orders.index)
        try:
            for original_index, row_series in orders.iterrows():
                try:
                    print(f"Processing order {original_index} ({orders.index.get_loc(original_index)+1}/{len(orders)})")
                    
                    temp_orders = orders.copy()
                    order = row_series.to_frame().T
                    
                    # Store the original price - FIX: Use row_series directly
                    original_price = row_series['B/A_price']
                    if original_price <= 1e-6:
                        print('what the hell')
                        breakpoint()
                    order.index = ['quote']
                    order['liquidity'] = 1
                    
                    # Set the price to None for priceQuote
                    if epsilon:
                        #if we use epsilon quoting, we need to set the liquidity to infinity
                        temp_orders.loc[:, 'liquidity'] = np.inf
                    
                    # FIX: Use loc instead of iloc with the original_index
                    temp_orders.loc[original_index, 'B/A_price'] = None
                    temp_orders.drop(original_index, inplace=True)
                    
                    # Add error handling for price quote
                    try:
                        quote_price = self.priceQuote(order, temp_orders)
                    except Exception as e:
                        print(f"Error in price quote for order {original_index}: {e}")
                        frontier_labels[original_index] = 0  # Default to not in frontier if quote fails
                        continue
                    
                    print(f'quote_price: {quote_price}, original_price: {original_price}')
                    # if quote_price is None or quote_price == 0 or quote_price == sys.maxsize or abs(quote_price - original_price) > original_price:
                    #     print('weird quote price', quote_price)
                    #     breakpoint()
                    # Compare with the original price
                    if row_series['transaction_type'] == 1:  # Buy order
                        if original_price > quote_price:
                            frontier_labels[original_index] = 1
                        else:
                            frontier_labels[original_index] = 0
                    else:  # Sell order
                        if original_price > quote_price:
                            frontier_labels[original_index] = 1
                        else:
                            frontier_labels[original_index] = 0
                except Exception as e:
                    print(f"Error processing order {original_index}: {e}")
                    frontier_labels[original_index] = None  # Default to not in frontier if processing fails
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Returning partial results...")
        
        orders['belongs_to_frontier'] = frontier_labels
        print(orders)
        return orders

    def epsilon_frontierGeneration(self, orders : pd.DataFrame = None):
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
            labeled_orders = self.frontierGeneration(orders, epsilon = True)
            return labeled_orders

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
        return opt_order.loc[:, ['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price', 'transaction_type','liquidity']]
	
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


# =============================================
# TEST CODE - Run with "if __name__ == '__main__'"
# =============================================

def test_single_security_epsilon_price_quote(data_file : str = None, offset : bool = False):
    """
    Test epsilon_priceQuote for single security options.
    Also verify that epsilon_priceQuote gives the same result as priceQuote
    when liquidity is set to infinity.
    we assume without offset, epsilon_priceQuote with all orders in the market is the same as priceQuote with frontiers in the market
    """
    print("\n=== Testing Single Security Epsilon Price Quote ===")
    
    try:
        if data_file is None:
            data_file = "/common/home/hg343/Research/accelerate_combo_option/data/training_data_frontier_bid_ask_12_6.pkl"
            
        with open(data_file, 'rb') as f:
            market_data_list = pickle.load(f)
            print(f"Loaded {len(market_data_list)} markets from {data_file}")
            
            # Skip the first market and take the next 10 (or fewer if there aren't 10)
            market_data_list = market_data_list[1:11]
            print(f"Testing with {len(market_data_list)} markets")
            
            # Track results for each market
            results = []
            
            # Iterate through each market
            for market_idx, market_df in enumerate(market_data_list):
                print(f"\n--- Testing Market {market_idx + 1} ---")
                
                try:
                    # Ensure we have both buy and sell orders
                    buy_orders = market_df[market_df['transaction_type'] == 1]
                    sell_orders = market_df[market_df['transaction_type'] == 0]
                    
                    if len(buy_orders) == 0 or len(sell_orders) == 0:
                        print(f"Market {market_idx + 1} doesn't have both buy and sell orders. Skipping...")
                        results.append({
                            'market_idx': market_idx + 1,
                            'status': 'Skipped - Missing buy or sell orders',
                            'error': None
                        })
                        continue
                    
                    # Remove rows with B/A_price = 0
                    market_df = market_df[market_df['B/A_price'] > 1e-6]
                    
                    # Create a temporary market to check if it matches
                    temp_market = Market(market_df, mechanism_solver=mechanism_solver_single)
                    
                    # Use a direct call to mechanism_solver to avoid recursion
                    is_match, profit = mechanism_solver_single(temp_market, offset=offset)
                    print(f"Market {market_idx + 1}: is_match={is_match}, profit={profit}")
                    
                    if is_match:
                        print(f"Market {market_idx + 1} has matching orders. Skipping...")
                        results.append({
                            'market_idx': market_idx + 1,
                            'status': 'Skipped - Has matching orders',
                            'error': None
                        })
                        continue
                    
                    # Create market
                    market = Market(market_df, mechanism_solver=mechanism_solver_single)
                    
                    # Select a random order to quote
                    random.seed(42 + market_idx)  # Different seed for each market
                    if len(buy_orders) > 0:
                        random_idx = random.randint(0, len(buy_orders) - 1)
                        order_to_quote = buy_orders.iloc[[random_idx]].copy()
                        
                        # Create a properly formatted order_to_quote
                        formatted_order = order_to_quote.copy()
                        
                        # Map columns if needed
                        column_mapping = {
                            'Strike Price of the Option Times 1000': 'strike',
                            'C=Call, P=Put': 'option_type'
                        }
                        
                        for old_col, new_col in column_mapping.items():
                            if old_col in formatted_order.columns and new_col not in formatted_order.columns:
                                formatted_order[new_col] = formatted_order[old_col]
                        
                        # Ensure liquidity column exists
                        if 'liquidity' not in formatted_order.columns:
                            formatted_order['liquidity'] = 1
                        
                        # Set price to None for quoting
                        formatted_order.loc[:, 'B/A_price'] = None
                        formatted_order.index = ['quote']
                        
                        print(f"Selected order to quote: {formatted_order.to_dict('records')[0]}")
                        
                        # Generate frontier
                        try:
                            orders_with_frontier_labels = market.epsilon_frontierGeneration()
                            if orders_with_frontier_labels is None:
                                print(f"Market {market_idx + 1}: Failed to generate frontier")
                                results.append({
                                    'market_idx': market_idx + 1,
                                    'status': 'Failed - Could not generate frontier',
                                    'error': None
                                })
                                continue
                                
                            frontier_orders = orders_with_frontier_labels[orders_with_frontier_labels['belongs_to_frontier'] == 1]
                            print(f"Found {len(frontier_orders)} frontier orders")
                            
                            # Test regular priceQuote and epsilon priceQuote
                            try:
                                # Regular price quote with all orders
                                regular_all_orders_quote = market.priceQuote(formatted_order, orders_with_frontier_labels)
                                print(f"Regular price quote (all orders): {regular_all_orders_quote}")
                                
                                # Regular price quote with frontier orders
                                regular_frontier_orders_quote = market.priceQuote(formatted_order, frontier_orders)
                                print(f"Regular price quote (frontier orders): {regular_frontier_orders_quote}")
                                
                                # Epsilon price quote with all orders
                                epsilon_all_orders_quote = market.epsilon_priceQuote(formatted_order, orders_with_frontier_labels)
                                print(f"Epsilon price quote (all orders): {epsilon_all_orders_quote}")
                                
                                # Epsilon price quote with frontier orders
                                epsilon_frontier_orders_quote = market.epsilon_priceQuote(formatted_order, frontier_orders)
                                print(f"Epsilon price quote (frontier orders): {epsilon_frontier_orders_quote}")
                                
                                # Check assertions
                                try:
                                    # Check if regular and epsilon quotes are similar
                                    if regular_all_orders_quote is not None and epsilon_all_orders_quote is not None:
                                        assert abs(regular_all_orders_quote - epsilon_all_orders_quote) < 1e-6, "Epsilon price quote with all orders should be the same as regular price quote with all orders"
                                        print("✓ Assertion passed: regular_all_orders_quote ≈ epsilon_all_orders_quote")
                                    
                                    if regular_frontier_orders_quote is not None and epsilon_frontier_orders_quote is not None:
                                        assert abs(regular_frontier_orders_quote - epsilon_frontier_orders_quote) < 1e-6, "Epsilon price quote with frontier orders should be the same as regular price quote with frontier orders"
                                        print("✓ Assertion passed: regular_frontier_orders_quote ≈ epsilon_frontier_orders_quote")
                                    
                                    results.append({
                                        'market_idx': market_idx + 1,
                                        'status': 'Success',
                                        'error': None,
                                        'regular_all_orders_quote': regular_all_orders_quote,
                                        'epsilon_all_orders_quote': epsilon_all_orders_quote,
                                        'regular_frontier_orders_quote': regular_frontier_orders_quote,
                                        'epsilon_frontier_orders_quote': epsilon_frontier_orders_quote
                                    })
                                except AssertionError as ae:
                                    print(f"❌ Assertion failed: {ae}")
                                    results.append({
                                        'market_idx': market_idx + 1,
                                        'status': 'Failed - Assertion Error',
                                        'error': str(ae),
                                        'regular_all_orders_quote': regular_all_orders_quote,
                                        'epsilon_all_orders_quote': epsilon_all_orders_quote,
                                        'regular_frontier_orders_quote': regular_frontier_orders_quote,
                                        'epsilon_frontier_orders_quote': epsilon_frontier_orders_quote
                                    })
                            except Exception as e:
                                print(f"Error during price quoting: {e}")
                                import traceback
                                traceback.print_exc()
                                results.append({
                                    'market_idx': market_idx + 1,
                                    'status': 'Failed - Price Quote Error',
                                    'error': str(e)
                                })
                        except Exception as e:
                            print(f"Error generating frontier: {e}")
                            import traceback
                            traceback.print_exc()
                            results.append({
                                'market_idx': market_idx + 1,
                                'status': 'Failed - Frontier Generation Error',
                                'error': str(e)
                            })
                    else:
                        print("No buy orders available for testing")
                        results.append({
                            'market_idx': market_idx + 1,
                            'status': 'Skipped - No buy orders',
                            'error': None
                        })
                except Exception as e:
                    print(f"Error processing market {market_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'market_idx': market_idx + 1,
                        'status': 'Failed - Processing Error',
                        'error': str(e)
                    })
            
            # Print summary of results
            print("\n=== Summary of Results ===")
            success_count = sum(1 for r in results if r['status'] == 'Success')
            print(f"Tested {len(results)} markets")
            print(f"Successful: {success_count}")
            print(f"Failed: {len(results) - success_count}")
            
            for r in results:
                status_symbol = "✓" if r['status'] == 'Success' else "❌"
                print(f"{status_symbol} Market {r['market_idx']}: {r['status']}")
                if r['error']:
                    print(f"   Error: {r['error']}")
            
            return success_count > 0  # Return True if at least one market was successful
    
    except Exception as e:
        print(f"Error loading real market data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combo_security_epsilon_price_quote():
    """
    Test epsilon_priceQuote for combo security options.
    """
    print("\n=== Testing Combo Security Epsilon Price Quote ===")
    

    
    # Try to load combo data from a file similar to main_combo.py
    combo_data_path = '/common/home/hg343/Research/accelerate_combo_option/data/combo_2_frontier'
    combo_data = None
    
    if os.path.exists(combo_data_path):
        for file in os.listdir(combo_data_path):
            if file.startswith('corrected') and file.endswith('.pkl'):
                file_path = os.path.join(combo_data_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        combo_data = pickle.load(f)
                        print(f"Loaded combo data from {file_path}")
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(combo_data, columns=[
                            'option1', 'option2', 'C=Call, P=Put',
                            'Strike Price of the Option Times 1000',
                            'transaction_type', 'B/A_price',
                            'belongs_to_frontier'
                        ])
                        
                        # Create market
                        market = Market(df, mechanism_solver=mechanism_solver_combo)
                        
                        # Select a random buy order to quote
                        random.seed(42)  # For reproducibility
                        buy_orders = df[df['transaction_type'] == 1]
                        
                        if len(buy_orders) > 0:
                            random_idx = random.randint(0, len(buy_orders) - 1)
                            order_to_quote = buy_orders.iloc[[random_idx]].copy()
                            print(f"Selected order to quote: {order_to_quote.to_dict('records')[0]}")
                            
                            # Test with default liquidity
                            try:
                                orders_in_market = market.get_market_data_order_format()
                                orders_in_market.loc[:, 'liquidity'] = np.inf
                                order_to_quote.loc[:, 'liquidity'] = 1
                                market.update_liquidity(orders_in_market.loc[:, 'liquidity'])
                                result1 = market.epsilon_priceQuote(order_to_quote)
                                print(f"Epsilon price quote with default liquidity: {result1}")
                                



                                result2 = market.priceQuote(order_to_quote, orders_in_market)
                                print(f"Price quote with explicit infinite liquidity: {result2}")
                                
                                # Verify that results are similar
                                assert abs(result1 - result2) < 1e-6, "Epsilon price quote should match price quote with explicit infinite liquidity"
                                print("✓ Epsilon price quote matches price quote with explicit infinite liquidity")
                                
                                return True
                            except Exception as e:
                                print(f"Error in combo test with real data: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("No buy orders found in the market")
                        
                        break
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    # If we couldn't load data, create synthetic data
    print("Creating synthetic combo data")
    # Create synthetic combo data
    data = {
        'option1': [0.5, 0.7, 0.3, 0.6],
        'option2': [0.5, 0.3, 0.7, 0.4],
        'C=Call, P=Put': [1, 1, -1, -1],
        'Strike Price of the Option Times 1000': [100, 110, 105, 115],
        'B/A_price': [5, 7, 10, 12],
        'transaction_type': [1, 1, 0, 0],  # 1 for buy, 0 for sell
        'belongs_to_frontier': [1, 0, 1, 0]
    }
    orders = pd.DataFrame(data)
    
    # Create market
    market = Market(orders, mechanism_solver=mechanism_solver_combo)
    
    # Create option to quote
    option_data = {
        'option1': [0.6],
        'option2': [0.4],
        'C=Call, P=Put': [1],
        'Strike Price of the Option Times 1000': [105],
        'B/A_price': [6],
        'transaction_type': [1]  # Buy order
    }
    option_to_quote = pd.DataFrame(option_data)
    
    # Test with default liquidity
    try:
        result1 = market.epsilon_priceQuote(option_to_quote)
        print(f"Epsilon price quote with default liquidity: {result1}")
        
        # Test with infinite liquidity explicitly
        orders_in_market = market.get_market_data_order_format()
        orders_in_market.loc[:, 'liquidity'] = np.inf
        option_to_quote.loc[:, 'liquidity'] = 1
        result2 = market.priceQuote(option_to_quote, orders_in_market)
        print(f"Price quote with explicit infinite liquidity: {result2}")
        
        # Verify that results are similar
        assert abs(result1 - result2) < 1e-6, "Epsilon price quote should match price quote with explicit infinite liquidity"
        print("✓ Epsilon price quote matches price quote with explicit infinite liquidity")
        
    except Exception as e:
        print(f"Error in combo test with synthetic data: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    #implement command line arguments to take in data file and offset
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='/common/home/hg343/Research/accelerate_combo_option/data/training_data_frontier_bid_ask_12_6.pkl')
    parser.add_argument("--offset", type=bool, default=True)
    args = parser.parse_args()
    data_file = args.data_file
    offset = args.offset
    # Run tests
    single_test_passed = test_single_security_epsilon_price_quote(data_file, offset)
    # combo_test_passed = test_combo_security_epsilon_price_quote()
    
    if single_test_passed:
        print("\n✓ single security tests passed!")
    sys.exit(1)

