from gurobipy import Model, GRB
import pandas as pd
import numpy as np
import sys
import timeit
from utils import Market 

def mechanism_solver_single(market: Market, offset : bool = True):
    '''
    Solve the mechanism solver given single security orders
    '''
    opt_l = offset # offset is 1 if offset is true, 0 if offset is false
    orders = market.get_market_data_order_format()
    assert 'C=Call, P=Put' in orders.columns, "C=Call, P=Put column not found in orders"
    assert 'Strike Price of the Option Times 1000' in orders.columns, "Strike Price of the Option Times 1000 column not found in orders"
    assert 'B/A_price' in orders.columns, "B/A_price column not found in orders"
    assert 'liquidity' in orders.columns, "liquidity column not found in orders"
    # Convert orders to numpy array for processing
    opt_buy = orders[orders.loc[:, 'transaction_type'] == 1]
    opt_sell = orders[orders.loc[:, 'transaction_type'] == 0]
    opt_buy_book = opt_buy[['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price']].to_numpy()
    opt_sell_book = opt_sell[['C=Call, P=Put', 'Strike Price of the Option Times 1000', 'B/A_price']].to_numpy()
    
    if len(opt_buy_book) == 0 or len(opt_sell_book) == 0:
        return None, None

    
    # Extract option data
    option_num_buy = len(opt_buy_book)
    option_num_sell = len(opt_sell_book)
    call_or_put = 0
    strike_price = 1
    premium = 2
    # Get unique strikes for constraints
    #strikes used for optimization 
    strikes = market.get_strikes()
    strikes.append(0)
    strikes.append(sys.maxsize)

    # Create optimization model
    model = Model("match")
    model.setParam('OutputFlag', False)
    
    # Decision variables - set upper bounds based on liquidity
    gamma = model.addVars(1, len(opt_buy_book), lb=0)  # sell to buys
    delta = model.addVars(1, len(opt_sell_book), lb=0)  # buy from asks
    # Set upper bounds based on liquidity
    for i in range(len(opt_buy_book)):
        assert 'liquidity' in opt_buy.columns, "liquidity column not found in opt_buy"
        liquidity = opt_buy.iloc[i]['liquidity']
        print('liquidity', liquidity)
        if np.isinf(liquidity):
            gamma[0, i].ub = 0.5#GRB.INFINITY
        else:
            gamma[0, i].ub = liquidity
    for i in range(len(opt_sell_book)):
        assert 'liquidity' in opt_sell.columns, "liquidity column not found in opt_sell"
        liquidity = opt_sell.iloc[i]['liquidity']
        if np.isinf(liquidity):
            delta[0, i].ub = 0.5#GRB.INFINITY
        else:
            delta[0, i].ub = liquidity
    # Add arbitrage constraints for each strike price
    if opt_l == 1:
        l = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        
    for strike in sorted(strikes):
        if opt_l == 1:
            model.addLConstr(
                sum(gamma[0,i]*max(opt_buy_book[i, call_or_put]*(strike-opt_buy_book[i, strike_price]), 0) for i in range(option_num_buy) ) - 
                sum(delta[0,i]*max(opt_sell_book[i, call_or_put]*(strike-opt_sell_book[i, strike_price]), 0) for i in range(option_num_sell)	) - 
                l[0,0], GRB.LESS_EQUAL, 0
            )
        else:
            model.addLConstr(
                sum(gamma[0,i]*max(opt_buy_book[i, call_or_put]*(strike-opt_buy_book[i, strike_price]), 0) for i in range(option_num_buy) ) - 
                sum(delta[0,i]*max(opt_sell_book[i, call_or_put]*(strike-opt_sell_book[i, strike_price]), 0) for i in range(option_num_sell)),
                GRB.LESS_EQUAL, 0
            )
    
    # Set objective function to maximize profit
    if opt_l == 1:
        model.setObjective(
            sum(gamma[0,i]*opt_buy_book[i, premium] for i in range(option_num_buy) ) - 
            sum(delta[0,i]*opt_sell_book[i, premium] for i in range(option_num_sell)),
            GRB.MAXIMIZE
        )
    else:
        model.setObjective(
            sum(gamma[0,i]*opt_buy_book[i, premium] for i in range(option_num_buy) ) - 
            sum(delta[0,i]*opt_sell_book[i, premium] for i in range(option_num_sell)),
            GRB.MAXIMIZE
        )
    
    # Solve the model
    model.optimize()
    
    # Print the decision variables that are non-zero with their index in the pandas dataframe and their value
    if model.status == GRB.OPTIMAL:
        profit = model.objVal
        isMatch = any(delta[0,i].x > 0 for i in range(len(delta))) or any(gamma[0,j].x > 0 for j in range(len(gamma)))
        
        # Print non-zero delta variables (buy decisions)
        for i in range(len(delta)):
            if delta[0,i].x > 1e-6:  # Use a small threshold to account for floating-point precision
                # Check if opt_sell_book is a DataFrame or a NumPy array
                if hasattr(opt_sell_book, 'index'):
                    # It's a DataFrame
                    print(f"Buy from sell_book[{i}]: {opt_sell_book.index[i]} - Amount: {delta[0,i].x:.4f}, Price: {opt_sell_book.iloc[i]['B/A_price']}")
                else:
                    # It's a NumPy array
                    print(f"Buy from sell_book[{i}] - Amount: {delta[0,i].x:.4f}, Price: {opt_sell_book[i, premium]}")
        
        # Print non-zero gamma variables (sell decisions)
        for j in range(len(gamma)):
            if gamma[0,j].x > 1e-6:  # Use a small threshold to account for floating-point precision
                # Check if opt_buy_book is a DataFrame or a NumPy array
                if hasattr(opt_buy_book, 'index'):
                    # It's a DataFrame
                    print(f"Sell to buy_book[{j}]: {opt_buy_book.index[j]} - Amount: {gamma[0,j].x:.4f}, Price: {opt_buy_book.iloc[j]['B/A_price']}")
                else:
                    # It's a NumPy array
                    print(f"Sell to buy_book[{j}] - Amount: {gamma[0,j].x:.4f}, Price: {opt_buy_book[j, premium]}")
        
        print(f"Total profit: {profit:.4f}")
        breakpoint()
        return isMatch, profit  # Match format from training.py
    else:
        print('model status is not optimal', model.status)
        breakpoint()

def mechanism_solver_combo(opt_buy_book : pd.DataFrame, opt_sell_book : pd.DataFrame, s1='S1', s2='S2', offset : bool = True, debug=0):
	'''
	opt_buy_book: pandas dataframe contains bid orders; specify whether code requires standarizing this variable
	opt_sell_book: pandas dataframe contains ask orders;
	s1: stock 1 name
	s2: stock 2 name
	order book: contains coefficients up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
    offset: whether to offset the price by the offset value
	debug: whether to debug
	'''
	buy_book_index = opt_buy_book.index
	sell_book_index = opt_sell_book.index
	
	# Extract liquidity values before converting to numpy arrays
	buy_liquidity = None
	sell_liquidity = None
	if 'liquidity' in opt_buy_book.columns:
		buy_liquidity = opt_buy_book['liquidity'].values
	if 'liquidity' in opt_sell_book.columns:
		sell_liquidity = opt_sell_book['liquidity'].values
	
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
		
		# Set upper bounds based on liquidity
		gamma = model.addVars(1, num_buy)  # sell to bid orders
		delta = model.addVars(1, num_sell)  # buy from ask orders
		
		# Check if liquidity column exists in the original dataframes before conversion to numpy
		if buy_liquidity is not None:
			for i in range(num_buy):
				if np.isinf(buy_liquidity[i]):
					gamma[0, i].ub = GRB.INFINITY
				else:
					gamma[0, i].ub = buy_liquidity[i]
		else:
			# Default upper bound of 1 if no liquidity specified
			for i in range(num_buy):
				gamma[0, i].ub = 1
				
		if sell_liquidity is not None:
			for i in range(num_sell):
				if np.isinf(sell_liquidity[i]):
					delta[0, i].ub = GRB.INFINITY
				else:
					delta[0, i].ub = sell_liquidity[i]
		else:
			# Default upper bound of 1 if no liquidity specified
			for i in range(num_sell):
				delta[0, i].ub = 1
		
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