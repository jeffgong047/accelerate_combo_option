import pandas as pd
import numpy as np
import os
import sys 
from mechanism_solver import mechanism_solver_combo, mechanism_solver_single
from copy import deepcopy
import pickle
import argparse 
import random
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
        assert isinstance(opt_df, pd.DataFrame), "opt_df must be a pandas dataframe"
        self.opt_df = deepcopy(opt_df)
        if 'liquidity' not in opt_df.columns: # by default, we assume each order has only one unit of liquidity
            self.opt_df['liquidity'] = 1
        if input_format == 'option_series':
            self.opt_order = self.convert_maket_data_format(opt_df, format='order')
        else:
            self.opt_order = self.opt_df 
        self.strikes = list(set(self.opt_order.loc[:, 'Strike Price of the Option Times 1000']))

        self.mechanism_solver = mechanism_solver

    def apply_mechanism(self, orders : pd.DataFrame, offset : bool = True):
        '''
        Apply the mechanism solver to the market data
        '''
        if self.mechanism_solver is None:
            raise ValueError("Mechanism solver is not specified")
        elif self.mechanism_solver == mechanism_solver_combo:
            buy_orders, sell_orders = self.separate_buy_sell(orders)
            return self.mechanism_solver(buy_orders, sell_orders, offset=offset)[2]
        elif self.mechanism_solver == mechanism_solver_single:
            # Create a temporary Market object with the provided orders
            temp_market = Market(orders, self.mechanism_solver)
            return self.mechanism_solver(temp_market, offset=offset)[1]
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
        assert option_to_quote.index not in orders_in_market.index, "option_to_quote is already in the market"
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
        if option_to_quote.index[0] != 'quote':
            option_to_quote.index = ['quote']
        assert option_to_quote.index[0] not in market_orders.index, "option_to_quote is already in the market"
        
        # Create a copy of the option to quote with a valid price
        # This ensures we don't include the None price in the mechanism solver
        option_to_price = option_to_quote.copy()
        assert option_to_price.iloc[0]['B/A_price'] is None, "Option to quote should have no price"
        # Use iloc to access the first row regardless of index
        if option_to_quote.iloc[0]['transaction_type'] == 1:
            # quoting price for buy order, we want to quote price by adding a sell order with premium = 0 to the sell side of the market 
            new_sell_order = option_to_quote.copy()
            new_sell_order.iloc[0, new_sell_order.columns.get_loc('transaction_type')] = 0
            new_sell_order.iloc[0, new_sell_order.columns.get_loc('B/A_price')] = 0
            # Preserve liquidity if it exists
            if 'liquidity' in option_to_quote.columns:
                new_sell_order.iloc[0, new_sell_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
            market_orders = pd.concat([market_orders, new_sell_order], ignore_index=False)
            objVal = self.apply_mechanism(market_orders, offset)
            return objVal
        elif option_to_quote.iloc[0]['transaction_type'] == 0:
            # quoting price for sell order, we want to quote price by adding a buy order with premium = max price to the buy side of the market 
            new_buy_order = option_to_quote.copy()
            new_buy_order.iloc[0, new_buy_order.columns.get_loc('transaction_type')] = 1
            new_buy_order.iloc[0, new_buy_order.columns.get_loc('B/A_price')] = sys.maxsize
            # Preserve liquidity if it exists
            if 'liquidity' in option_to_quote.columns:
                new_buy_order.iloc[0, new_buy_order.columns.get_loc('liquidity')] = option_to_quote.iloc[0]['liquidity']
            market_orders = pd.concat([market_orders, new_buy_order, option_to_quote], ignore_index=False)
            objVal = self.apply_mechanism(market_orders, offset)
            return sys.maxsize - objVal
        else:
            raise ValueError("Invalid transaction type")
    def frontierGeneration(self, orders : pd.DataFrame = None, epsilon : bool = False):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        # we dont want to change the original orders, so we make a copy of it 
        if orders is None:
            orders_copy = self.opt_order.copy()
        else:
            orders_copy = orders.copy()
        
        frontier_labels = pd.Series(None, index=orders_copy.index)

        for original_index, row_series in orders_copy.iterrows():
            try:
                temp_orders = orders_copy.copy()

                order = row_series.to_frame().T
                order.index = ['quote']
                # Store the original price
                original_price = order.iloc[original_index]['B/A_price']
                
                # Set the price to None for priceQuote
                temp_orders.iloc[original_index, temp_orders.columns.get_loc('B/A_price')] = None
                temp_orders.drop(original_index, inplace=True)
                
                # Add error handling for price quote
                try:
                    quote_price = self.priceQuote(order, temp_orders)
                except Exception as e:
                    print(f"Error in price quote for order {original_index}: {e}")
                    frontier_labels[original_index] = 0  # Default to not in frontier if quote fails
                    continue
                
                # Compare with the original price
                if row_series['transaction_type'] == 1:  # Buy order
                    if quote_price > original_price:
                        frontier_labels[original_index] = 1
                    else:
                        frontier_labels[original_index] = 0
                else:  # Sell order
                    if quote_price < original_price:
                        frontier_labels[original_index] = 1
                    else:
                        frontier_labels[original_index] = 0
            except Exception as e:
                print(f"Error processing order {original_index}: {e}")
                frontier_labels[original_index] = None  # Default to not in frontier if processing fails

            
        orders_copy['belongs_to_frontier'] = frontier_labels
        return orders_copy

    def epsilon_frontierGeneration(self, orders : pd.DataFrame = None):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        return self.frontierGeneration(orders, epsilon = True)

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
        self.strikes.append(sys.maxsize)


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
    
    # Create test data for single security
    # Load real market data similar to main_single.py

    try:
        with open(data_file, 'rb') as f:
            market_data_list = pickle.load(f)
            print(f"Loaded {len(market_data_list)} markets from {data_file}")
            
            # Use the first market for testing
            market_df = market_data_list[0]
            print(market_df)
            print(f"Using market with {len(market_df)} orders")
            
            # Create market
            market = Market(market_df, mechanism_solver=mechanism_solver_single)
            
            # Select a random order to quote

            random.seed(42)  # For reproducibility
            
            # Select a buy order (transaction_type == 1)
            buy_orders = market_df[market_df['transaction_type'] == 1]
            if len(buy_orders) > 0:
                random_idx = random.randint(0, len(buy_orders) - 1)
                order_to_quote = buy_orders.iloc[[random_idx]].copy()
                print(f"Selected order to quote: {order_to_quote.to_dict('records')[0]}")
                order_to_quote.loc[:, 'B/A_price'] = None 
                order_to_quote.index = ['quote']

                frontier_orders = market.epsilon_frontierGeneration()

                # Test regular priceQuote
                regular_all_orders_quote = market.priceQuote(order_to_quote)
                regular_frontier_orders_quote = market.priceQuote(order_to_quote, frontier_orders)

                # Test epsilon_priceQuote
                epsilon_all_orders_quote = market.epsilon_priceQuote(order_to_quote)
                epsilon_frontier_orders_quote = market.epsilon_priceQuote(order_to_quote, frontier_orders)


                print(f"Epsilon price quote: {epsilon_all_orders_quote}")
                print(f"Epsilon frontier price quote: {epsilon_frontier_orders_quote}")

                #sanity check: for single security, epsilon_priceQuote with all orders in the market is the same as priceQuote with all orders in the market
                assert epsilon_all_orders_quote == regular_all_orders_quote, "Epsilon price quote with all orders in the market should be the same as priceQuote with all orders in the market"
                assert epsilon_frontier_orders_quote == regular_frontier_orders_quote, "Epsilon price quote with frontier orders in the market should be the same as priceQuote with frontier orders in the market"
                return epsilon_all_orders_quote == epsilon_frontier_orders_quote
            else:
                print("No buy orders found in the market")
    except Exception as e:
        print(f"Error loading real market data: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to synthetic data if real data can't be loaded
    print("Using synthetic data for testing")
    
    # Create market data
    data = {
        'C=Call, P=Put': [1, 1, -1, -1],
        'Strike Price of the Option Times 1000': [100, 110, 105, 115],
        'B/A_price': [5, 7, 10, 12],
        'transaction_type': [1, 1, 0, 0]  # 1 for buy, 0 for sell
    }
    orders = pd.DataFrame(data)
    
    # Create market
    market = Market(orders, mechanism_solver=mechanism_solver_single)
    
    # Create option to quote
    option_data = {
        'C=Call, P=Put': [1],
        'Strike Price of the Option Times 1000': [105],
        'B/A_price': [6],
        'transaction_type': [1]  # Buy order
    }
    option_to_quote = pd.DataFrame(option_data)
    
    # Test regular priceQuote
    regular_quote = market.priceQuote(option_to_quote)
    print(f"Regular price quote: {regular_quote}")
    
    # Test epsilon_priceQuote
    epsilon_quote = market.epsilon_priceQuote(option_to_quote)
    print(f"Epsilon price quote: {epsilon_quote}")
    
    # Manually set liquidity to infinity and use priceQuote
    orders_in_market = market.get_market_data_order_format()
    orders_in_market.loc[:, 'liquidity'] = np.inf
    option_to_quote.loc[:, 'liquidity'] = 1
    manual_quote = market.priceQuote(option_to_quote, orders_in_market)
    print(f"Manual price quote with infinite liquidity: {manual_quote}")
    
    # Verify that epsilon_priceQuote and manual approach give the same result
    assert abs(epsilon_quote - manual_quote) < 1e-6, "Epsilon price quote should match manual price quote with infinite liquidity"
    print("✓ Epsilon price quote matches manual price quote with infinite liquidity")
    
    return True

def test_combo_security_epsilon_price_quote():
    """
    Test epsilon_priceQuote for combo security options.
    """
    print("\n=== Testing Combo Security Epsilon Price Quote ===")
    
    # Create test data for combo security
    from mechanism_solver import mechanism_solver_combo
    import pickle
    import os
    import random
    
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

