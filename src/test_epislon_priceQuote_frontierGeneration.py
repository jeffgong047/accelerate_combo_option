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
from market import Market
# Add this at the beginning of your script
def signal_handler(sig, frame):
    print('\nProgram interrupted by user. Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)




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
            if market_data_list is not list:
                market_data_list = [market_data_list]
            print(f"Testing with {len(market_data_list)} markets")
            
            # Track results for each market
            results = []
            
            # Iterate through each market
            for market_idx, market_df in enumerate(market_data_list):
                print(f"\n--- Testing Market {market_idx + 1} ---")
                market_attrs = market_df.attrs.copy()
                print(market_attrs)
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
                    temp_market.update_liquidity(pd.Series(np.ones(len(market_df)), index=market_df.index))
                    orders = temp_market.get_market_data_order_format()
                    # Use a direct call to mechanism_solver to avoid recursion
                    is_match, profit = temp_market.apply_mechanism(orders)
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
                                        'epsilon_frontier_orders_quote': epsilon_frontier_orders_quote,
                                        'market_attrs': market_attrs
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
            with open('results.pkl', 'wb') as f:
                pickle.dump(results, f)
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
                        # sanity check on implementation of combo_mechanism_solver
                        market = Market(df, mechanism_solver=mechanism_solver_combo)
                        liquidity = pd.Series(np.ones(len(df))*10, index=df.index)
                        df = df.assign(liquidity=liquidity)
                        isMatch1, profit1 = market.apply_mechanism(df)
                        opt_buy_book = df[df['transaction_type'] == 1].to_numpy()
                        opt_sell_book = df[df['transaction_type'] == 0].to_numpy()
                        _, _,profit2 = synthetic_combo_match_mip(opt_buy_book, opt_sell_book)
                        isMatch2 = profit2 > 0
                        print(f"isMatch1: {isMatch1}, profit1: {profit1}")
                        print(f"isMatch2: {isMatch2}, profit2: {profit2}")  
                        # Select a random buy order to quote
                        random.seed(42)  # For reproducibility
                        buy_orders = df[df['transaction_type'] == 1]
                        
                        if len(buy_orders) > 0:
                            random_idx = random.randint(0, len(buy_orders) - 1)
                            order_to_quote = buy_orders.iloc[[random_idx]].copy()
                            order_to_quote_times_10 = buy_orders.iloc[[random_idx]].copy()
                            order_to_quote_times_10.loc[:, 'B/A_price'] = order_to_quote_times_10.loc[:, 'B/A_price'] * 10
                            order_to_quote_times_10.loc[:, 'option1'] = order_to_quote_times_10.loc[:, 'option1'] * 10
                            order_to_quote_times_10.loc[:, 'option2'] = order_to_quote_times_10.loc[:, 'option2'] * 10
                            order_to_quote_times_10000 = buy_orders.iloc[[random_idx]].copy()
                            order_to_quote_times_10000.loc[:, 'B/A_price'] = order_to_quote_times_10000.loc[:, 'B/A_price'] * 10000
                            order_to_quote_times_10000.loc[:, 'option1'] = order_to_quote_times_10000.loc[:, 'option1'] * 10000
                            order_to_quote_times_10000.loc[:, 'option2'] = order_to_quote_times_10000.loc[:, 'option2'] * 10000
                            print(f"Selected order to quote: {order_to_quote.to_dict('records')[0]}")
                            # Test with default liquidity
                            try:
                            #     orders_in_market = market.get_market_data_order_format()
                            #     orders_in_market.loc[:, 'liquidity'] = 1
                            #     order_to_quote.loc[:, 'liquidity'] = 1
                            #     market.update_liquidity(orders_in_market.loc[:, 'liquidity'])
                            #     profit_liquidity_1 = market.priceQuote(order_to_quote)
                            #     profit_liquidity_1_times_10 = market.priceQuote(order_to_quote_times_10)
                            #     profit_liquidity_1_times_10000 = market.priceQuote(order_to_quote_times_10000)

                            #     orders_in_market.loc[:, 'liquidity'] = 10
                            #     order_to_quote.loc[:, 'liquidity'] = 1
                            #     market.update_liquidity(orders_in_market.loc[:, 'liquidity'])
                            #     profit_liquidity_10 = market.priceQuote(order_to_quote)
                            #     profit_liquidity_10_times_10 = market.priceQuote(order_to_quote_times_10)
                            #     profit_liquidity_10_times_10000 = market.priceQuote(order_to_quote_times_10000)

                            #     profit_liquidity_infinity = market.epsilon_priceQuote(order_to_quote)
                            #     profit_liquidity_infinity_times_10 = market.epsilon_priceQuote(order_to_quote_times_10)
                            #     profit_liquidity_infinity_times_10000 = market.epsilon_priceQuote(order_to_quote_times_10000)
                                
                            #     print(f"price quote with default liquidity 1: {profit_liquidity_1}")
                            #     print(f"price quote with default liquidity 1, ten times original order: {profit_liquidity_1_times_10}")
                            #     print(f"price quote with default liquidity 1, ten thousand times original order: {profit_liquidity_1_times_10000}")
                            #     print(f"price quote with default liquidity 10: {profit_liquidity_10}")
                            #     print(f"price quote with default liquidity 10, ten times original order: {profit_liquidity_10_times_10}")
                            #     print(f"price quote with default liquidity 10, ten thousand times original order: {profit_liquidity_10_times_10000}")
                            #     print(f"price quote with default liquidity infinity: {profit_liquidity_infinity}")
                            #     print(f"price quote with default liquidity infinity, ten times original order: {profit_liquidity_infinity_times_10}")
                            #     print(f"price quote with default liquidity infinity, ten thousand times original order: {profit_liquidity_infinity_times_10000}")
                            #     breakpoint()

                            #     # Verify that results are similar
                            #     assert abs(profit_liquidity_1 - profit_liquidity_infinity) < 1e-6, "Epsilon price quote should match price quote with explicit infinite liquidity"
                            #     # 
                            #     print("✓ Epsilon price quote matches price quote    with explicit infinite liquidity")
                                # similar to @test_single_security_epsilon_price_quote, i want to test whether epsilon_priceQuote with all orders in the market is the same as priceQuote with frontiers in the market 
                                orders_with_frontier_labels = market.epsilon_frontierGeneration()
                                if orders_with_frontier_labels is None:
                                    raise Exception("Failed to generate frontier")
                                    continue
                                print(f"orders_with_frontier_labels: {orders_with_frontier_labels}")
                                breakpoint()  
                                frontier_orders = orders_with_frontier_labels[orders_with_frontier_labels['belongs_to_frontier'] == 1]
                                print(f"Found {len(frontier_orders)} frontier orders")
                                
                                # Test regular priceQuote and epsilon priceQuote
                                try:
                                    # Regular price quote with all orders
                                    regular_all_orders_quote = market.priceQuote(order_to_quote, orders_with_frontier_labels)
                                    print(f"Regular price quote (all orders): {regular_all_orders_quote}")
                                    
                                    # Regular price quote with frontier orders
                                    regular_frontier_orders_quote = market.priceQuote(order_to_quote, frontier_orders)
                                    print(f"Regular price quote (frontier orders): {regular_frontier_orders_quote}")
                                    
                                    # Epsilon price quote with all orders
                                    epsilon_all_orders_quote = market.epsilon_priceQuote(order_to_quote, orders_with_frontier_labels)
                                    print(f"Epsilon price quote (all orders): {epsilon_all_orders_quote}")
                                    
                                    # Epsilon price quote with frontier orders
                                    epsilon_frontier_orders_quote = market.epsilon_priceQuote(order_to_quote, frontier_orders)
                                    print(f"Epsilon price quote (frontier orders): {epsilon_frontier_orders_quote}")
                                    
                                    # Check assertions
                                    # if regular_all_orders_quote is not None and epsilon_all_orders_quote is not None:
                                    #     assert abs(regular_all_orders_quote - epsilon_all_orders_quote) < 1e-6, "Epsilon price quote with all orders should be the same as regular price quote with all orders"
                                    #     print("✓ Assertion passed: regular_all_orders_quote ≈ epsilon_all_orders_quote")
                                    
                                    # if regular_frontier_orders_quote is not None and epsilon_frontier_orders_quote is not None:
                                    #     assert abs(regular_frontier_orders_quote - epsilon_frontier_orders_quote) < 1e-6, "Epsilon price quote with frontier orders should be the same as regular price quote with frontier orders"
                                    #     print("✓ Assertion passed: regular_frontier_orders_quote ≈ epsilon_frontier_orders_quote")
                                except Exception as e:
                                    print(f"Error during price quoting: {e}")
                                    import traceback
                                    traceback.print_exc()



                                breakpoint()
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
    
    


if __name__ == "__main__":
    #implement command line arguments to take in data file and offset
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='/common/home/hg343/Research/accelerate_combo_option/data/training_data_frontier_bid_ask_12_6.pkl')
    parser.add_argument("--offset", type=bool, default=True)
    args = parser.parse_args()
    data_file = args.data_file
    offset = args.offset
    # Run tests
    # single_test_passed = test_single_security_epsilon_price_quote(data_file, offset)
    combo_test_passed = test_combo_security_epsilon_price_quote()
    
    # if single_test_passed:
    #     print("\n✓ single security tests passed!")
    sys.exit(1)

