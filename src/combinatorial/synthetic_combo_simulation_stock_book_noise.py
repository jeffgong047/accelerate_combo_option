import pdb
import pickle

import numpy as np
import random
import math
import os.path
from gen_synthetic_constrained_combo_options import *
from synthetic_combo_mip_match import *
from gurobipy import *
import timeit
from copy import deepcopy

def synthetic_combo_frontier_generation(opt_buy_book, opt_sell_book, s1='S1', s2='S2', debug=0):
	num_buy, num_sell, num_stock = len(opt_buy_book), len(opt_sell_book), len(opt_buy_book[0])-4
	# add initial constraints
	f_constraints = []
	f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], np.zeros((num_stock, 1))))-opt_buy_book[:, -3]), 0))
	f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_buy_book[:, -3]), 0))
	g_constraints = []
	g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], np.zeros((num_stock, 1))))-opt_sell_book[:, -3]), 0))
	g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_sell_book[:, -3]), 0))
	# coeff up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
	opt_sell_book_frontier_labels = np.zeros(len(opt_sell_book))
	opt_buy_book_frontier_labels = np.zeros(len(opt_buy_book))
	opt_buy_book_holder = deepcopy(opt_buy_book)
	opt_sell_book_holder = deepcopy(opt_sell_book)
	# find frontier bids:
	for option_index in range(len(opt_buy_book_holder)):
		sub_obj = 1
		#add N+1 option of the buy option to seller side and set ask price = 0
		opt_buy_book = deepcopy(opt_buy_book_holder)
		bid = opt_buy_book[option_index][5]
		copied_opt_sell = opt_buy_book[option_index]
		#lets assume we are only handling two option case
		copied_opt_sell[5] = 0
		print(opt_buy_book.shape)
		opt_buy_book = np.concatenate([opt_buy_book, np.expand_dims(copied_opt_sell, axis= 0 )],axis=0)
		print(opt_buy_book.shape)
		try:
			# prime problem
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
		if model.objVal <= bid:
			#add it to frontiers
			opt_buy_book_frontier_labels[option_index] = 1
		else:
			assert opt_buy_book_frontier_labels[option_index] == 0

	for option_index in range(len(opt_sell_book)):
		sub_obj = 1
		#add sell option to buy side of the market and set b_(M+1) price  = 10^6
		opt_sell_book = deepcopy(opt_sell_book_holder)
		ask = opt_buy_book[option_index][5]
		copied_opt_buy = opt_sell_book[option_index]
		#lets assume we are only handling two option case
		copied_opt_sell[5] = 1e6
		opt_buy_book = np.concatenate([opt_sell_book, np.expand_dims(copied_opt_buy, axis = 0)] ,axis=0)
		try:
			# prime problem
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
		if model.objVal >= ask:
			opt_sell_book_frontier_labels[option_index] = 1
		else:
			assert opt_sell_book_frontier_labels[option_index] == 0

	frontier_buy_book = np.concatenate([opt_buy_book_holder, np.expand_dims(opt_buy_book_frontier_labels, axis = 1)],axis = 1)
	frontier_sell_book = np.concatenate([opt_sell_book_holder, np.expand_dims(opt_sell_book_frontier_labels, axis = 1)], axis = 1)
	return np.concatenate([frontier_buy_book, frontier_sell_book], axis = 0)




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


SIM_NUM = 20
# constr_rev = np.load('combo_vary_stock_size_constrained_pairs.npy')
frontier_option_training_data_list = []
for NUM_STOCK in [2]:#, 4, 8, 12, 16, 20]: #12, 16, 20
	for BOOK_SIZE in [20]: #, 100, 150, 200, 250, 300, 350, 400]:
		for NOISE in [2**-4]:#, 2**-6, 2**-5, 2**-4, 2**-3]:
			for i in range(1, SIM_NUM+1):
				# get order book
				filename = 'book/STOCK_{}_SEED_{}_book.npy'.format(NUM_STOCK, i)
				#filename = 'book_constrained/STOCK_{}_SEED_{}_book.npy'.format(NUM_STOCK, i)
				if os.path.isfile(filename):
					opt_book = np.load(filename)
				else:
					opt_book = gen_synthetic_combo_options(NUM_ST=NUM_STOCK, NUM_ORDER=1024, SEED=i)
					np.save(filename, opt_book)

				opt_book_1 = opt_book[:BOOK_SIZE]
				opt_buy_book, opt_sell_book = add_noise_orderbook(opt_book_1, NOISE)
				print('#####Generating {} with size {} and noise {}#####'.format(filename, BOOK_SIZE, NOISE))
				frontier_option_label = synthetic_combo_frontier_generation(opt_buy_book, opt_sell_book, debug=1)
				frontier_option_training_data_list.append(frontier_option_label)
				# print('#####Matching {} with size {} and noise {}: num_constr = {} and profit = {}#####'.format(filename, BOOK_SIZE, NOISE, num_iter, round(profit,2)))
				# constr_rev[int(NUM_STOCK/4), i-1, 0] = num_iter
				# constr_rev[int(NUM_STOCK/4), i-1, 1] = profit
				# np.save('combo_vary_stock_size_constrained_pairs.npy', constr_rev)

with open( 'combo_frontier_training.pkl','wb') as f:
	pickle.dump(frontier_option_training_data_list,f)

pdb.set_trace()