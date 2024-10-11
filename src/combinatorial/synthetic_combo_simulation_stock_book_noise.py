import pdb
import numpy as np
import random
import math
import os.path
from gen_synthetic_constrained_combo_options import *
from synthetic_combo_mip_match import *
from gurobipy import *

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
for NUM_STOCK in [2, 4, 8, 12, 16, 20]: #12, 16, 20
	for BOOK_SIZE in [50, 100, 150, 200, 250, 300, 350, 400]:
		for NOISE in [2**-7, 2**-6, 2**-5, 2**-4, 2**-3]:
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
				_, num_iter, profit = synthetic_combo_match_mip(opt_buy_book, opt_sell_book, debug=1)
				print('#####Matching {} with size {} and noise {}: num_constr = {} and profit = {}#####'.format(filename, BOOK_SIZE, NOISE, num_iter, round(profit,2)))
				# constr_rev[int(NUM_STOCK/4), i-1, 0] = num_iter
				# constr_rev[int(NUM_STOCK/4), i-1, 1] = profit
				# np.save('combo_vary_stock_size_constrained_pairs.npy', constr_rev)

				opt_book_2 = opt_book[BOOK_SIZE:BOOK_SIZE*2]
				opt_buy_book, opt_sell_book = add_noise_orderbook(opt_book_2, NOISE)
				print('#####Generating {} with size {} and noise {}#####'.format(filename, BOOK_SIZE, NOISE))
				_, num_iter, profit = synthetic_combo_match_mip(opt_buy_book, opt_sell_book, debug=1)
				print('#####Matching {} with size {} and noise {}: num_constr = {} and profit = {}#####'.format(filename, BOOK_SIZE, NOISE, num_iter, round(profit,2)))
				# constr_rev[int(NUM_STOCK/4), SIM_NUM+i-1, 0] = num_iter
				# constr_rev[int(NUM_STOCK/4), SIM_NUM+i-1, 1] = profit
				# np.save('combo_vary_stock_size_constrained_pairs.npy', constr_rev)
pdb.set_trace()