import pdb
import sys
import numpy as np
import random
import math
import timeit
from gurobipy import *
import pandas as pd
from copy import deepcopy

def synthetic_combo_match_mip(opt_buy_book : pd.DataFrame, opt_sell_book : pd.DataFrame, s1='S1', s2='S2', debug=0):
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
	matched_stock_indices = {'buy_book_index': None, 'sell_book_index': None}
	matched_stock_indices['buy_book_index'] = [buy_book_index[i] for i in range(len(gamma)) if gamma[0, i].x > 0]
	matched_stock_indices['sell_book_index'] = [sell_book_index[i] for i in range(len(delta)) if delta[0, i].x > 0]
	return time, model.NumConstrs, model.objVal, isMatch, matched_stock_indices