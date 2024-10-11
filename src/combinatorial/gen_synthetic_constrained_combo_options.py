import pdb
import sys
import os
import numpy as np
import random
import itertools
import pandas as pd
import datetime, time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gurobipy import *

def gen_pairs(stock_list, last_level, this_level, SEED=1):
	if last_level < 2:
		new_res = list(itertools.combinations(stock_list, 2))
		np.save('pairs_{}.npy'.format(this_level), new_res)
	else:
		size = len(stock_list)
		res = np.load('pairs_{}.npy'.format(last_level)).tolist()
		pool = [list(pair) for pair in itertools.combinations(stock_list, 2) if list(pair) not in res]
		stock = set() # include all stocks
		while len(stock) < size:
			new_res, stock = [], set()
			new_res = res + random.sample(pool, k=size-len(res))
			for pair in new_res:
				stock.add(pair[0])
				stock.add(pair[1])
		np.save('pairs_{}.npy'.format(this_level), new_res)
	return new_res

def gen_synthetic_combo_options(NUM_ST=2, NUM_ORDER=100, SEED=1):
	COEFF_CAP = 10
	random.seed(SEED)
	if NUM_ST == 2: #1
		stock_list = ['KO', 'MCD']
		pair_list = np.load('data/pairs_2.npy')
	elif NUM_ST == 4: #6
		stock_list = ['DIS', 'KO', 'MCD', 'NKE'] #6
		pair_list = np.load('data/pairs_4.npy')
	elif NUM_ST == 8: #28
		stock_list = ['DIS', 'IBM', 'KO', 'MCD', 'MSFT', 'NKE', 'WMT', 'XOM']
		pair_list = np.load('data/pairs_8.npy')
	elif NUM_ST == 12: #66
		stock_list = ['AXP', 'DIS', 'HD', 'IBM', 'JNJ', 'KO', 'MCD', 'MSFT', 'NKE', 'VZ', 'WMT', 'XOM']
		pair_list = np.load('data/pairs_12.npy')
	elif NUM_ST == 16: #120
		stock_list = ['AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MSFT', 'NKE', 'PG', 'VZ', 'WMT', 'XOM']
		pair_list = np.load('data/pairs_16.npy')
	elif NUM_ST == 20: #190
		stock_list = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM']
		pair_list = np.load('data/pairs_20.npy')
	
	# generate synthetic combo options
	# coeff up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
	opt_book = np.zeros((NUM_ORDER, len(stock_list)+4))
	num_options = 0
	buy_flag = 0
	print(pair_list.tolist())
	while num_options < NUM_ORDER:
		# randomly select stocks
		pair = random.sample(pair_list.tolist(), k=1)
		index = [stock_list.index(pair[0][0]), stock_list.index(pair[0][1])]
		# print(stock_list[index[0]], stock_list[index[1]])
		# generate combinatorial contracts from single security options
		opt1_book = np.load('data/'+stock_list[index[0]]+'.npy')
		opt2_book = np.load('data/'+stock_list[index[1]]+'.npy')
		opt1 = random.choice(opt1_book)
		opt2 = random.choice(opt2_book)
		def coprime(a,b):
			p, q = a, b
			while q != 0:
				p, q = q, p%q
			return int(a/p), int(b/p)
		coeff1, coeff2 = coprime(random.choice(range(1, COEFF_CAP)), random.choice(range(1, COEFF_CAP)))
		# print(coeff1, stock_list[index[0]], opt1[0], opt1[1], opt1[2], opt1[3])
		# print(coeff2, stock_list[index[1]], opt2[0], opt2[1], opt2[2], opt2[3])

		# coeff up to len(stock_list); call_put; strike; buy_sell; bid_ask
		opt = np.zeros(len(stock_list)+4)
		strike = coeff1*opt1[0]*opt1[1]+coeff2*opt2[0]*opt2[1]
		call_put = 1 if strike > 0 else -1
		opt[index[0]] = coeff1 if opt1[0]==call_put else -1*coeff1
		opt[index[1]] = coeff2 if opt2[0]==call_put else -1*coeff2
		opt[len(stock_list)] = call_put
		opt[len(stock_list)+1] = np.abs(strike)
		# print('The combo option is {}({}{} {}{}, {})'.format('C' if opt[len(stock_list)]==1 else 'P', opt[index[0]], stock_list[index[0]], opt[index[1]], stock_list[index[1]], opt[len(stock_list)+1]))
		# pdb.set_trace()
		# decide price based on single security options
		option_num = len(opt1_book) + len(opt2_book)
		strikes_1 = list(set(opt1_book[:, 1]))
		strikes_1.append(0)
		strikes_1.append(sys.maxsize)
		strikes_2 = list(set(opt2_book[:, 1]))
		strikes_2.append(0)
		strikes_2.append(sys.maxsize)

		if buy_flag or random.random() < 0.5:
			# print('finding the highest bid')
			opt[len(stock_list)+2] = 1
			bid = 0
			try:
				model = Model("gen_match")
				model.setParam('OutputFlag', False)
				# decision variables
				gamma = model.addVars(1, option_num) #sell
				delta = model.addVars(1, option_num) #buy
				D = model.addVars(1, 1, lb=0, ub=1)
				L = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
				# add constraints at every knot value
				for K_1 in sorted(strikes_1):
					for K_2 in sorted(strikes_2):
						target_buy = D[0,0]*max(opt[len(stock_list)]*(opt[index[0]]*K_1+opt[index[1]]*K_2-opt[len(stock_list)+1]), 0)
						opt1_buy_sum = sum(delta[0,i]*max(opt1_book[i][0]*(K_1-opt1_book[i][1]), 0) for i in range(len(opt1_book)))
						opt1_sell_sum = sum(gamma[0,i]*max(opt1_book[i][0]*(K_1-opt1_book[i][1]), 0) for i in range(len(opt1_book)) if opt1_book[i][2] > 0)
						opt2_buy_sum = sum(delta[0,len(opt1_book)+i]*max(opt2_book[i][0]*(K_2-opt2_book[i][1]), 0) for i in range(len(opt2_book)))
						opt2_sell_sum = sum(gamma[0,len(opt1_book)+i]*max(opt2_book[i][0]*(K_2-opt2_book[i][1]), 0) for i in range(len(opt2_book)) if opt2_book[i][2] > 0)
						model.addLConstr(opt1_sell_sum+opt2_sell_sum-opt1_buy_sum-opt2_buy_sum-target_buy-L[0,0], GRB.LESS_EQUAL, 0)
				# define obj
				opt1_expense = sum(delta[0,i]*opt1_book[i][3] for i in range(len(opt1_book)))
				opt1_gain = sum(gamma[0,i]*opt1_book[i][2] for i in range(len(opt1_book)))
				opt2_expense = sum(delta[0,len(opt1_book)+i]*opt2_book[i][3] for i in range(len(opt2_book)))
				opt2_gain = sum(gamma[0,len(opt1_book)+i]*opt2_book[i][2] for i in range(len(opt2_book)))
				target_expense = D[0,0]*bid
				model.setObjective(opt1_gain+opt2_gain-opt1_expense-opt2_expense-target_expense-L[0,0], GRB.MAXIMIZE)
				model.optimize()
				# for v in model.getVars():
				# 	print('%s %g' % (v.varName, v.x))
				expense = 0
				for i in range(0, option_num):
					if not (delta[0, i].x == 1 and gamma[0, i].x == 1):
						if round(gamma[0, i].x,4) != 0:
							if i < len(opt1_book):
								# print('Sell {} {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), stock_list[index[0]], opt1_book[i][0], opt1_book[i][1], opt1_book[i][2]))
								expense = expense+gamma[0,i].x*opt1_book[i][2]
							else:
								# print('Sell {} {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), stock_list[index[1]], opt2_book[i-len(opt1_book)][0], opt2_book[i-len(opt1_book)][1], opt2_book[i-len(opt1_book)][2]))
								expense = expense+gamma[0,i].x*opt2_book[i-len(opt1_book)][2]
						if round(delta[0, i].x,4) != 0:
							if i < len(opt1_book):
								# print('Buy {} {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), stock_list[index[0]], opt1_book[i][0], opt1_book[i][1], opt1_book[i][3]))
								expense = expense-delta[0, i].x*opt1_book[i][3]
							else:
								# print('Buy {} {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), stock_list[index[1]], opt2_book[i-len(opt1_book)][0], opt2_book[i-len(opt1_book)][1], opt2_book[i-len(opt1_book)][3]))
								expense = expense-delta[0, i].x*opt2_book[i-len(opt1_book)][3]				
				# print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), round(L[0,0].x,2)))
				# pdb.set_trace()
				bid = round(model.objVal,2)
				# print('#################The highest bid is {}#################'.format(bid))
				if bid > 0:
					opt[len(stock_list)+3] = bid
			except GurobiError as e:
				print('Error code ' + str(e.errno) + ": " + str(e))
			except AttributeError:
				print('Encountered an attribute error')
		else:
			# print('finding the lowest ask')
			opt[len(stock_list)+2] = 0
			ask = 10000
			try:
				model = Model("gen_match")
				model.setParam('OutputFlag', False)
				# decision variables
				gamma = model.addVars(1, option_num) #sell
				delta = model.addVars(1, option_num) #buy
				G = model.addVars(1, 1, lb=0, ub=1)
				L = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
				# add constraints at every knot value
				for K_1 in sorted(strikes_1):
					for K_2 in sorted(strikes_2):
						target_sell = G[0,0]*max(opt[len(stock_list)]*(opt[index[0]]*K_1+opt[index[1]]*K_2-opt[len(stock_list)+1]), 0)
						opt1_buy_sum = sum(delta[0,i]*max(opt1_book[i][0]*(K_1-opt1_book[i][1]), 0) for i in range(len(opt1_book)))
						opt1_sell_sum = sum(gamma[0,i]*max(opt1_book[i][0]*(K_1-opt1_book[i][1]), 0) for i in range(len(opt1_book)) if opt1_book[i][2] > 0)
						opt2_buy_sum = sum(delta[0,len(opt1_book)+i]*max(opt2_book[i][0]*(K_2-opt2_book[i][1]), 0) for i in range(len(opt2_book)))
						opt2_sell_sum = sum(gamma[0,len(opt1_book)+i]*max(opt2_book[i][0]*(K_2-opt2_book[i][1]), 0) for i in range(len(opt2_book)) if opt2_book[i][2] > 0)
						model.addLConstr(target_sell+opt1_sell_sum+opt2_sell_sum-opt1_buy_sum-opt2_buy_sum-L[0,0], GRB.LESS_EQUAL, 0)
				# define obj
				opt1_expense = sum(delta[0,i]*opt1_book[i][3] for i in range(len(opt1_book)))
				opt1_gain = sum(gamma[0,i]*opt1_book[i][2] for i in range(len(opt1_book)))
				opt2_expense = sum(delta[0,len(opt1_book)+i]*opt2_book[i][3] for i in range(len(opt2_book)))
				opt2_gain = sum(gamma[0,len(opt1_book)+i]*opt2_book[i][2] for i in range(len(opt2_book)))
				target_gain = G[0,0]*ask
				model.setObjective(target_gain+opt1_gain+opt2_gain-opt1_expense-opt2_expense-L[0,0], GRB.MAXIMIZE)
				model.optimize()
				# for v in model.getVars():
				# 	print('%s %g' % (v.varName, v.x))
				expense = 0
				for i in range(0, option_num):
					if not (delta[0, i].x == 1 and gamma[0, i].x == 1):
						if round(gamma[0, i].x,4) != 0:
							if i < len(opt1_book):
								# print('Sell {} {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), stock_list[index[0]], opt1_book[i][0], opt1_book[i][1], opt1_book[i][2]))
								expense = expense+gamma[0,i].x*opt1_book[i][2]
							else:
								# print('Sell {} {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), stock_list[index[1]], opt2_book[i-len(opt1_book)][0], opt2_book[i-len(opt1_book)][1], opt2_book[i-len(opt1_book)][2]))
								expense = expense+gamma[0,i].x*opt2_book[i-len(opt1_book)][2]
						if round(delta[0, i].x,4) != 0:
							if i < len(opt1_book):
								# print('Buy {} {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), stock_list[index[0]], opt1_book[i][0], opt1_book[i][1], opt1_book[i][3]))
								expense = expense-delta[0, i].x*opt1_book[i][3]
							else:
								# print('Buy {} {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), stock_list[index[1]], opt2_book[i-len(opt1_book)][0], opt2_book[i-len(opt1_book)][1], opt2_book[i-len(opt1_book)][3]))
								expense = expense-delta[0, i].x*opt2_book[i-len(opt1_book)][3]				
				# print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), round(L[0,0].x,2)))
				# pdb.set_trace()
				ask = round(ask-model.objVal,2)
				# print('#################The lowest ask is {}#################'.format(ask))
				if ask < 10000:
					opt[len(stock_list)+3] = ask
			except GurobiError as e:
				print('Error code ' + str(e.errno) + ": " + str(e))
			except AttributeError:
				print('Encountered an attribute error')
		# add the generated option to book
		if opt[len(stock_list)+3] > 0:
			buy_flag = 0
			opt_book[num_options] = opt
			print('#{}: The order is {} combo option {}({}{} {}{}, {}) at {}'.format(num_options, 'Buy' if opt[len(stock_list)+2] else 'Sell', 'C' if opt[len(stock_list)]==1 else 'P', opt[index[0]], stock_list[index[0]], opt[index[1]], stock_list[index[1]], opt[len(stock_list)+1], opt[len(stock_list)+3]))
			num_options = num_options+1
		else:
			buy_flag = 1
	return opt_book