import pandas as pd
import numpy as np
import os
import sys 
from mechanism_solver import mechanism_solver_combo, mechanism_solver_single

'''
The script provides a class financial_option_market that implements the following main functionalities:
1. epsilon_priceQuote: generate the frontier of options with epsilon price quote
2. epsilon_frontierGeneration: generate the frontier of options with epsilon price quote and constraints
'''


class Market:
    def __init__(self, opt_df: pd.DataFrame, mechanism_solver=None,input_format = None):
        '''
        The initialization should ensure the object could process all the functions in the class 
        opt_df: pandas dataframe of the market data columns: 
        columns: list of columns in the dataframe 
        input_format: the format of the market data, either 'option_series' or 'order format'   
        mechanism_solver: the mechanism solver to compute the profit of the given market. 
        If one wants to customized mechanism solver, one just need to ensure the input to the mechanism solver takes in orders in pandas dataframe format, and returns profit as first output.
        
        Strikes: market will prepare unique strikes for orders in the market and 0, infinity for constraints
        '''
        assert isinstance(opt_df, pd.DataFrame), "opt_df must be a pandas dataframe"
        self.opt_df = opt_df 
        if input_format == 'option_series':
            self.opt_order = self.convert_market_data_format(opt_df, format='order')
        else:
            self.opt_order = self.opt_df 
        self.strikes = list(set(self.opt_order.loc[:, 'Strike Price of the Option Times 1000']))
        self.strikes.append(0)
        self.strikes.append(sys.maxsize)

        self.mechanism_solver = mechanism_solver

    def apply_mechanism(self, orders : pd.DataFrame):
        '''
        Apply the mechanism solver to the market data
        '''
        if self.mechanism_solver is None:
            raise ValueError("Mechanism solver is not specified")
        elif self.mechanism_solver == mechanism_solver_combo:
            buy_orders, sell_orders = self.separate_buy_sell(orders)
            return self.mechanism_solver(buy_orders, sell_orders)[2]
        elif self.mechanism_solver == mechanism_solver_single:
            return self.mechanism_solver(orders)[1]
        else:
            return self.mechanism_solver(orders)[0]
    
    def priceQuote(self, order : pd.DataFrame, offset: bool = True, epsilon: bool = False):
        '''
        Generate the price of of givne input order w.r.t orders in the market
        '''
        market_orders = self.get_market_data_order_format()
        if order.loc[0, 'transaction_type'] == 1:
            # quoting price for buy order, we want to quote price by adding a sell order with premium = 0 to the sell side of the market 
    
            new_sell_order = order.copy()
            new_sell_order.loc[:, 'transaction_type'] = 0
            new_sell_order.loc[:, 'B/A_price'] = 0
            market_orders = pd.concat([market_orders, order, new_sell_order], ignore_index=False)
            objVal = self.apply_mechanism(market_orders, offset, epsilon)
            return objVal
        elif order.loc[0, 'transaction_type'] == 0:
            # quoting price for sell order, we want to quote price by adding a buy order with premium = max price to the buy side of the market 
            new_buy_order = order.copy()
            new_buy_order.loc[:, 'transaction_type'] = 1
            new_buy_order.loc[:, 'B/A_price'] = sys.maxsize
            market_orders = pd.concat([market_orders, new_buy_order, order], ignore_index=False)
            objVal = self.apply_mechanism(market_orders, offset, epsilon)
            return sys.maxsize - objVal
        else:
            raise ValueError("Invalid transaction type")
    def frontierGeneration(self, orders : pd.DataFrame, epsilon: bool = False):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        # we dont want to change the original orders, so we make a copy of it 
        orders_copy = orders.copy()
        frontier_labels = pd.Series(None, index=orders_copy.index)
        for index, row_series in orders_copy.iterrows():
            temp_orders = orders_copy.copy()
            original_index = index 
            order = row_series.to_frame().T
            order.index = [original_index]
            temp_orders.drop(index, inplace=True)
            quote_price = self.priceQuote(order, temp_orders, epsilon)
            if row_series['belongs_to_frontier'] == 1:
                # this is bid order 
                if quote_price > row_series['B/A_price']:
                    frontier_labels[original_index] = 1
                else:
                    frontier_labels[original_index] = 0
            else:
                if quote_price < row_series['B/A_price']:
                    frontier_labels[original_index] = 1
                else:
                    frontier_labels[original_index] = 0
        orders['belongs_to_frontier'] = frontier_labels

        return orders
    
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

    def separate_buy_sell(self):
        '''
        return [sell book , buy book]
        '''
        return self.opt_df[self.opt_df['transaction_type'] == 1], self.opt_df[self.opt_df['transaction_type'] == 0]
    def get_strikes(self):
        return self.strikes 
    
    def get_market_data_raw(self):
        return self.opt_df.copy()
    
    def get_market_data_order_format(self):
        return self.opt_order.copy()
	
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






