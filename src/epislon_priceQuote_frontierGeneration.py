import pandas as pd
import numpy as np
import os
import sys 
from gurobipy import Model, GRB
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
    
    def priceQuote(self, orders : pd.DataFrame, offset: bool = False):
        '''
        Generate the price of of givne input order w.r.t orders in the market
        '''
        market_orders = self.get_market_data_order_format()
        if offset:
            pass 
        else:
            pass 

    def frontierGeneration(self, orders : pd.DataFrame):
        '''
        Generate the frontier of options with epsilon price quote and constraints
        '''
        pass


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






def mechanism_solver_single(orders : pd.DataFrame):
    '''
    Solve the mechanism solver given single security orders
    '''

    
    # Convert orders to numpy array for processing
    buy_book = orders[orders['transaction_type'] == 1]
    sell_book = orders[orders['transaction_type'] == 0]
    
    if len(buy_book) == 0 or len(sell_book) == 0:
        return 0, 0, 0, 0, 0  # Return zeros if no matching possible
    
    # Extract option data
    option_num = len(orders)
    opt = orders[['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price', 'transaction_type']].to_numpy()
    
    # Get unique strikes for constraints
    strikes = orders['Strike Price of the Option Times 1000'].unique()
    
    # Create optimization model
    model = Model("match")
    model.setParam('OutputFlag', False)
    
    # Decision variables
    gamma = model.addVars(1, option_num, lb=0, ub=1)  # sell to buys
    delta = model.addVars(1, option_num, lb=0, ub=1)  # buy from asks
    
    # Add arbitrage constraints for each strike price
    opt_l = 1  # Use slack variable for constraints
    if opt_l == 1:
        l = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        
    for strike in sorted(strikes):
        if opt_l == 1:
            model.addLConstr(
                sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 3] == 1) - 
                sum(delta[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 3] == 0) - 
                l[0,0], GRB.LESS_EQUAL, 0
            )
        else:
            model.addLConstr(
                sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 3] == 1) - 
                sum(delta[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 3] == 0),
                GRB.LESS_EQUAL, 0
            )
    
    # Set objective function to maximize profit
    if opt_l == 1:
        model.setObjective(
            sum(gamma[0,i]*opt[i, 2] for i in range(option_num) if opt[i, 3] == 1) - 
            sum(delta[0,i]*opt[i, 2] for i in range(option_num) if opt[i, 3] == 0) - 
            l[0,0], GRB.MAXIMIZE
        )
    else:
        model.setObjective(
            sum(gamma[0,i]*opt[i, 2] for i in range(option_num) if opt[i, 3] == 1) - 
            sum(delta[0,i]*opt[i, 2] for i in range(option_num) if opt[i, 3] == 0),
            GRB.MAXIMIZE
        )
    
    # Solve the model
    model.optimize()
    
    # Return results
    if model.status == GRB.OPTIMAL:
        profit = model.objVal
        return profit, 0, profit, 0, 0  # Match format from training.py
    else:
        return 0, 0, 0, 0, 0  # Return zeros if no solution found


def mechanism_solver_combo(opt_buy_book : pd.DataFrame, opt_sell_book : pd.DataFrame, s1='S1', s2='S2', debug=0):
	'''
	opt_buy_book: pandas dataframe contains bid orders; specify whether code requires standarizing this variable
	opt_sell_book: pandas dataframe contains ask orders;
	s1: stock 1 name
	s2: stock 2 name
	order book: contains coefficients up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)

	debug: whether to debug
	'''
	buy_book_index = opt_buy_book.index
	sell_book_index = opt_sell_book.index
	sorted_columns_order = ['option1', 'option2','C=Call, P=Put',
                'Strike Price of the Option Times 1000',
                'transaction_type', 'B/A_price']
	opt_buy_book = opt_buy_book[sorted_columns_order].to_numpy()
	opt_sell_book = opt_sell_book[sorted_columns_order].to_numpy()
	num_buy, num_sell, num_stock = len(opt_buy_book), len(opt_sell_book), len(opt_buy_book[0])-4
	# add initial constraints

	f_constraints = []
	f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], np.zeros((num_stock, 1))))-opt_buy_book[:, -3]), 0))
	f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_buy_book[:, -3]), 0))
	g_constraints = []
	g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], np.zeros((num_stock, 1))))-opt_sell_book[:, -3]), 0))
	g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_sell_book[:, -3]), 0))
	sub_obj = 1

	try:
		# prime problemsys
		model = Model("match")
		model.setParam('OutputFlag', False)
		gamma = model.addVars(1, num_buy, ub=1) #sell to bid orders
		delta = model.addVars(1, num_sell, ub=1) #buy from ask orders
		L = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
		# constraint of 0
		buy_sum = sum(delta[0,i]*g_constraints[0][i] for i in range(num_sell))
		sell_sum = sum(gamma[0,i]*f_constraints[0][i] for i in range(num_buy))
		model.addLConstr(sell_sum-buy_sum-L[0,0], GRB.LESS_EQUAL, 0)
		# define obj
		expense = sum(delta[0,i]*opt_sell_book[i, -1] for i in range(num_sell))
		gain = sum(gamma[0,i]*opt_buy_book[i, -1] for i in range(num_buy))
		model.setObjective(gain-expense-L[0,0], GRB.MAXIMIZE)

		# sub problem
		sub_model = Model("sub_match")
		sub_model.setParam('OutputFlag', False)
		M = 1000000
		s = sub_model.addVars(1, num_stock)
		f = sub_model.addVars(1, num_buy, lb=-GRB.INFINITY)
		g = sub_model.addVars(1, num_sell)
		I = sub_model.addVars(1, num_buy, vtype=GRB.BINARY)
		for i in range(num_sell):
			sub_model.addLConstr(opt_sell_book[i, -4]*(sum(opt_sell_book[i, j]*s[0,j] for j in range(num_stock))-opt_sell_book[i, -3])-g[0, i], GRB.LESS_EQUAL, 0)
		for i in range(num_buy):
			sub_model.addLConstr(opt_buy_book[i, -4]*(sum(opt_buy_book[i, j]*s[0,j] for j in range(num_stock))-opt_buy_book[i, -3])+M*(1-I[0, i])-f[0, i], GRB.GREATER_EQUAL, 0)
			sub_model.addLConstr(M*I[0, i]-f[0, i], GRB.GREATER_EQUAL, 0)
			sub_model.addLConstr(opt_buy_book[i, -4]*(sum(opt_buy_book[i, j]*s[0,j] for j in range(num_stock))-opt_buy_book[i, -3])+M*(1-I[0, i]), GRB.GREATER_EQUAL, 0)
			sub_model.addLConstr(opt_buy_book[i, -4]*(sum(opt_buy_book[i, j]*s[0,j] for j in range(num_stock))-opt_buy_book[i, -3])-M*I[0, i], GRB.LESS_EQUAL, 0)

		it = 0
		start = timeit.default_timer()
		while sub_obj > 0.0005:
			# add newly generated constraint
			buy_sum_new = sum(delta[0,i]*g_constraints[-1][i] for i in range(num_sell))
			sell_sum_new = sum(gamma[0,i]*f_constraints[-1][i] for i in range(num_buy))
			model.addLConstr(sell_sum_new-buy_sum_new-L[0,0], GRB.LESS_EQUAL, 0)
			model.optimize()
			# for v in model.getVars():
			# 	print('%s %g' % (v.varName, v.x))
			# save decision variables from prime problem
			gamma_val = np.array([max(gamma[0, i].x, 0) for i in range(num_buy)])
			delta_val = np.array([max(delta[0, i].x, 0) for i in range(num_sell)])
			L_val = L[0,0].x
			if debug == 2:
				print(gamma_val)
				print(delta_val)
				print(L_val)

			# define sub obj
			sub_model.setObjective(sum(gamma_val[i]*f[0, i] for i in range(num_buy))-sum(delta_val[i]*g[0, i] for i in range(num_sell))-L_val, GRB.MAXIMIZE)
			sub_model.optimize()
			# for v in sub_model.getVars():
			# 	print('%s %g' % (v.varName, v.x))
			if debug > 0:
				if it % 100 == 0:
					print([s[0, i].x for i in range(num_stock)])
					print('{}: objective is {} > 0'.format(it, sub_model.objVal))
				if debug == 2:
					for i in range(num_buy):
						print('I:', I[0, i].x)
						print('f:', f[0, i].x)
					for i in range(num_sell):
						print('g:', g[0, i].x)
			# save decision variables from sub problem			
			f_constraints.append(np.array([f[0, i].x for i in range(num_buy)]))
			g_constraints.append(np.array([g[0, i].x for i in range(num_sell)]))
			sub_obj = sub_model.objVal
			it += 1

		stop = timeit.default_timer()
		time = stop - start
		# print matching result
		if debug == 1:
			revenue = 0
			for i in range(num_buy):
				if gamma[0, i].x > 0:
					revenue += gamma[0,i].x * opt_buy_book[i, -1]
					print('Sell {} to {}({}{}+{}{},{}) at bid price {}'.format(round(gamma[0,i].x, 4), 'C' if opt_buy_book[i, -4]==1 else 'P', \
						opt_buy_book[i, 0], s1, opt_buy_book[i, 1], s2, opt_buy_book[i, -3], opt_buy_book[i, -1]))
			for i in range(num_sell):
				if delta[0, i].x > 0:
					revenue -= delta[0,i].x * opt_sell_book[i, -1]
					print('Buy {} from {}({}{}+{}{},{}) at ask price {}'.format(round(delta[0,i].x, 4), 'C' if opt_sell_book[i, -4]==1 else 'P', \
						opt_sell_book[i, 0], s1, opt_sell_book[i, 1], s2, opt_sell_book[i, -3], opt_sell_book[i, -1]))
			print('Revenue at T0 is {}; L is {}; Objective is {} = {}'.format(round(revenue,2), round(L[0,0].x, 2), round(revenue-L[0,0].x, 2), round(model.objVal, 2)))
	except GurobiError as e:
		print('Error code ' + str(e.errno) + ": " + str(e))
	except AttributeError:
		print('Encountered an attribute error')
	
	if model is None:
		return time, 0, 0
	isMatch = any(delta[0,i].x > 0 for i in range(len(delta))) or any(gamma[0,j].x > 0 for j in range(len(gamma)))
	matched_stock = {'buy_book_index': None, 'sell_book_index': None}
	matched_stock['buy_book_index'] = [buy_book_index[i] for i in range(len(gamma)) if gamma[0, i].x > 0]
	matched_stock['sell_book_index'] = [sell_book_index[i] for i in range(len(delta)) if delta[0, i].x > 0]
	return time, model.NumConstrs, model.objVal, isMatch, matched_stock