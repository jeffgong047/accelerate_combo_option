import pdb
import pickle
import argparse
import pandas as pd
import numpy as np
import random
import math
import os.path
from combinatorial.gen_synthetic_combo_options import gen_synthetic_combo_options
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
from gurobipy import *
import timeit
from copy import deepcopy
from tqdm import tqdm
# Run in a separate process with timeout
from multiprocessing import Process, Queue, Pool
import multiprocessing as mp 
import queue
import traceback
from contextlib import contextmanager
import signal
import sys
from multiprocessing import Pool, TimeoutError 
import itertools
import wandb
import os 


# # Add this function to parse command line arguments
# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Process stock options.')
#     parser.add_argument('--num_stocks', type=int, default=2, help='Number of stocks to process (default: 3)')
#     parser.add_argument('--market_size', type=int, default=50, help='Number of orders in the market')
#     parser.add_argument('--offset', type=bool, default=False, help='Whether to allow offset for liability in the optimization')
#     parser.add_argument('--wandb_project', type=str, default='expediating_comb_financial_market_matching', help='Wandb project name')
#     parser.add_argument('--num_orders', type=int, default=5000, help='number of orders in the orderbook')
#     parser.add_argument('--noise', type=float, default=2**-2, help='noise level in the orderbook')
#     parser.add_argument('--stock_combo', type=str, default=None, help='Comma-separated list of stock symbols to use (e.g. "AAPL,MSFT")')
#     parser.add_argument('--seed', type=int, default=1, help='Random seed for generating order books')
#     parser.add_argument('--output_dir', type=str, default=None, help='Output directory for generated order books')
#     return parser.parse_args()

# # Move the main execution code inside if __name__ == '__main__':
# args = parse_arguments()



# Set wandb API key programmatically
os.environ["WANDB_API_KEY"] = "d1cb0d609d7b64218fe82a45a54e57f47e2d26da"

try:
    wandb.login()  # This will now use the API key we just set
except wandb.errors.AuthError:
    print("Could not authenticate with wandb. Invalid API key")
    sys.exit(1)

def signal_handler(signum, frame):
    print("Ctrl+C received. Terminating processes...")
    if 'pool' in globals():
        pool.terminate()
        pool.join()
    sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def synthetic_combo_frontier_generation(original_orders_df: pd.DataFrame, s1='S1', s2='S2', offset = False, debug=0):
    '''
    opt_buy_book_holder: pandas dataframe contains bid orders; regardless of input; transformed to column order: ['option1', 'option2', 'C=Call, P=Put','Strike Price of the Option Times 1000', 'transaction_type','B/A_price']
    opt_sell_book_holder: pandas dataframe contains ask orders; ...
    s1: stock 1 name
    s2: stock 2 name
    opt_l: whether have offset or budget on liability in the optimization
    debug: whether to debug
    order book: contains coefficients up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
    '''
    #first check if is match and provide frontier if not:
    breakpoint()
    original_opt_buy_book = original_orders_df.where(original_orders_df['transaction_type'] == 1).dropna()
    original_opt_sell_book = original_orders_df.where(original_orders_df['transaction_type'] == 0).dropna()
    original_opt_buy_book = original_opt_buy_book[['option1', 'option2', 'C=Call, P=Put','Strike Price of the Option Times 1000', 'transaction_type','B/A_price']]
    original_opt_sell_book = original_opt_sell_book[['option1', 'option2', 'C=Call, P=Put','Strike Price of the Option Times 1000', 'transaction_type','B/A_price']]
    _, num_iter, profit , isMatch, matched_stock = synthetic_combo_match_mip(deepcopy(original_opt_buy_book), deepcopy(original_opt_sell_book), offset= offset, debug=0)
    quote_price = pd.Series(index=original_opt_buy_book.index)
    buy_book_index = original_opt_buy_book.index
    sell_book_index = original_opt_sell_book.index
    if isMatch: 
        #remove the matched 
        #check no match exist after removal of matched.
        # if 'buy_book_index' in matched_stock and 'sell_book_index' in matched_stock:
        #     remove_buy_book = matched_stock['buy_book_index']
        #     remove_sell_book = matched_stock['sell_book_index']
        #     filtered_index = lambda filter_index, all_index: [idx for idx in all_index if idx not in filter_index]
        #     opt_buy_book_filtered_index =  filtered_index(remove_buy_book, buy_book_index)
        #     opt_sell_book_filtered_index = filtered_index(remove_sell_book, sell_book_index) 
        #     opt_buy_book_holder = deepcopy(original_opt_buy_book.loc[opt_buy_book_filtered_index])
        #     opt_sell_book_holder = deepcopy(original_opt_sell_book.loc[opt_sell_book_filtered_index])
        #     _, num_iter, profit , isMatch_new, matched_stock= synthetic_combo_match_mip(opt_buy_book_holder, opt_sell_book_holder, debug=0)
        assert 'buy_book_index' in matched_stock and 'sell_book_index' in matched_stock
        if 'buy_book_index' in matched_stock and 'sell_book_index' in matched_stock:
            remove_buy_book = set(matched_stock['buy_book_index'])  # Convert to set for faster lookup
            remove_sell_book = set(matched_stock['sell_book_index'])
            
            # Use pandas index operations instead
            opt_buy_book_filtered_index = buy_book_index.difference(remove_buy_book)
            opt_sell_book_filtered_index = sell_book_index.difference(remove_sell_book)
            print(buy_book_index, remove_buy_book)
            print(sell_book_index, remove_sell_book)
            opt_buy_book_holder = deepcopy(original_opt_buy_book.loc[opt_buy_book_filtered_index])
            opt_sell_book_holder = deepcopy(original_opt_sell_book.loc[opt_sell_book_filtered_index])
            print(len(opt_buy_book_holder))
            print(len(opt_sell_book_holder))
            if len(opt_buy_book_holder) == 0 or len(opt_sell_book_holder) == 0:
                return None 
            _, num_iter, profit , isMatch_new, matched_stock= synthetic_combo_match_mip(opt_buy_book_holder.copy(), opt_sell_book_holder.copy(), debug=0)

        if isMatch_new:
            return None 

    else:
        opt_buy_book_holder = deepcopy(original_opt_buy_book)
        opt_sell_book_holder = deepcopy(original_opt_sell_book)

    num_buy_holder, num_sell_holder, num_stock = len(opt_buy_book_holder), len(opt_sell_book_holder), len(opt_buy_book_holder.columns)-4
    # coeff up to len(stock_list); call/put; strike; buy/sell; price (bid/ask)
    opt_sell_book_frontier_labels = np.zeros(len(opt_sell_book_holder))
    opt_buy_book_frontier_labels = np.zeros(len(opt_buy_book_holder))
    
    # find frontier bids:
    for option_index in tqdm(range(len(opt_buy_book_holder)), desc='Checking buy side options'):
        option_df_index = opt_buy_book_holder.index[option_index]
        sub_obj = 1
        #add N+1 option of the buy option to seller side and set ask price = 0
        opt_buy_book = deepcopy(opt_buy_book_holder).to_numpy()
        opt_sell_book = deepcopy(opt_sell_book_holder).to_numpy()
        # what is inside of buy book is actually ask orders 
        bid = opt_buy_book[option_index][5]
        copied_opt_sell = deepcopy(opt_buy_book[option_index])
        #lets assume we are only handling two option case
        copied_opt_sell[5] = 0
        opt_sell_book = np.concatenate([opt_sell_book, np.expand_dims(copied_opt_sell, axis= 0 )],axis=0)
        num_buy = len(opt_buy_book)
        num_sell = len(opt_sell_book)
        # add initial constraints
        f_constraints = []
        f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], np.zeros((num_stock, 1))))-opt_buy_book[:, -3]), 0))
        f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_buy_book[:, -3]), 0))
        g_constraints = []
        g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], np.zeros((num_stock, 1))))-opt_sell_book[:, -3]), 0))
        g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_sell_book[:, -3]), 0))
        try:
            # prime problem
            model = Model("match")
            model.setParam('OutputFlag', False)
            gamma = model.addVars(1, num_buy, ub=1) #sell to bid orders
            delta = model.addVars(1, num_sell, ub=1) #buy from ask orders
            if offset:
                L = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            # constraint of 0
            buy_sum = sum(delta[0,i]*g_constraints[0][i] for i in range(num_sell))
            sell_sum = sum(gamma[0,i]*f_constraints[0][i] for i in range(num_buy))
            if offset:
                model.addLConstr(sell_sum-buy_sum-L[0,0], GRB.LESS_EQUAL, 0)
            else:
                model.addLConstr(sell_sum-buy_sum, GRB.LESS_EQUAL, 0)
            expense = sum(delta[0,i]*opt_sell_book[i, -1] for i in range(num_sell))
            gain = sum(gamma[0,i]*opt_buy_book[i, -1] for i in range(num_buy))
            if offset:
                model.setObjective(gain-expense-L[0,0], GRB.MAXIMIZE)
            else:
                model.setObjective(gain-expense, GRB.MAXIMIZE)

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
                if offset:
                    model.addLConstr(sell_sum_new-buy_sum_new-L[0,0], GRB.LESS_EQUAL, 0)
                else:
                    model.addLConstr(sell_sum_new-buy_sum_new, GRB.LESS_EQUAL, 0)
                model.optimize()
                # for v in model.getVars():
                # 	print('%s %g' % (v.varName, v.x))
                # save decision variables from prime problem
                gamma_val = np.array([max(gamma[0, i].x, 0) for i in range(num_buy)])
                delta_val = np.array([max(delta[0, i].x, 0) for i in range(num_sell)])
                if offset:
                    L_val = L[0,0].x
                if debug == 2:
                    print(gamma_val)
                    print(delta_val)
                    print(L_val)

                # define sub obj
                if offset:
                    sub_model.setObjective(sum(gamma_val[i]*f[0, i] for i in range(num_buy))-sum(delta_val[i]*g[0, i] for i in range(num_sell))-L_val, GRB.MAXIMIZE)
                else:
                    sub_model.setObjective(sum(gamma_val[i]*f[0, i] for i in range(num_buy))-sum(delta_val[i]*g[0, i] for i in range(num_sell)), GRB.MAXIMIZE)
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
            #add it to frontier s
            print(f'original bid: {bid},quoted bid: {model.objVal}')
            opt_buy_book_frontier_labels[option_index] = 1
        else:
            assert opt_buy_book_frontier_labels[option_index] == 0
        quote_price[option_df_index] = model.objVal
    for option_index in tqdm(range(len(opt_sell_book_holder)), desc='Checking sell side options'):
        option_df_index = opt_sell_book_holder.index[option_index]
        sub_obj = 1
        #add sell option to buy side of the market and set b_(M+1) price  = 10^6
        opt_sell_book = deepcopy(opt_sell_book_holder).to_numpy()
        opt_buy_book = deepcopy(opt_buy_book_holder).to_numpy()
        ask = opt_sell_book[option_index][5]
        copied_opt_buy = deepcopy(opt_sell_book[option_index])
        opt_sell_book = np.delete(opt_sell_book, option_index, axis=0)
        #lets assume we are only handling two option case
        copied_opt_buy[5] = 1e6
        opt_buy_book = np.concatenate([opt_buy_book, np.expand_dims(copied_opt_buy, axis = 0)] ,axis=0)
        num_buy = len(opt_buy_book)
        num_sell = len(opt_sell_book)
        # add initial constraints
        f_constraints = []
        f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], np.zeros((num_stock, 1))))-opt_buy_book[:, -3]), 0))
        f_constraints.append(np.maximum(opt_buy_book[:, -4]*(np.concatenate(np.matmul(opt_buy_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_buy_book[:, -3]), 0))
        g_constraints = []
        g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], np.zeros((num_stock, 1))))-opt_sell_book[:, -3]), 0))
        g_constraints.append(np.maximum(opt_sell_book[:, -4]*(np.concatenate(np.matmul(opt_sell_book[:, :-4], sys.maxsize*np.ones((num_stock, 1))))-opt_sell_book[:, -3]), 0))
        try:
            # prime problem
            model = Model("match")
            model.setParam('OutputFlag', False)
            gamma = model.addVars(1, num_buy, ub=1) #sell to bid orders
            delta = model.addVars(1, num_sell, ub=1) #buy from ask orders
            if offset:
                L = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            # constraint of 0
            buy_sum = sum(delta[0,i]*g_constraints[0][i] for i in range(num_sell))
            sell_sum = sum(gamma[0,i]*f_constraints[0][i] for i in range(num_buy))
            if offset:
                model.addLConstr(sell_sum-buy_sum-L[0,0], GRB.LESS_EQUAL, 0)
            else:
                model.addLConstr(sell_sum-buy_sum, GRB.LESS_EQUAL, 0)
            # define obj
            expense = sum(delta[0,i]*opt_sell_book[i, -1] for i in range(num_sell))
            gain = sum(gamma[0,i]*opt_buy_book[i, -1] for i in range(num_buy))
            if offset:
                model.setObjective(gain-expense-L[0,0], GRB.MAXIMIZE)
            else:
                model.setObjective(gain-expense, GRB.MAXIMIZE)

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
                if offset:   
                    model.addLConstr(sell_sum_new-buy_sum_new-L[0,0], GRB.LESS_EQUAL, 0)
                else:
                    model.addLConstr(sell_sum_new-buy_sum_new, GRB.LESS_EQUAL, 0)
                model.optimize()
                # for v in model.getVars():
                # 	print('%s %g' % (v.varName, v.x))
                # save decision variables from prime problem
                gamma_val = np.array([max(gamma[0, i].x, 0) for i in range(num_buy)])
                delta_val = np.array([max(delta[0, i].x, 0) for i in range(num_sell)])
                if offset:
                    L_val = L[0,0].x
                if debug == 2:
                    print(gamma_val)
                    print(delta_val)
                    print(L_val)

                # define sub obj
                if offset:
                    sub_model.setObjective(sum(gamma_val[i]*f[0, i] for i in range(num_buy))-sum(delta_val[i]*g[0, i] for i in range(num_sell))-L_val, GRB.MAXIMIZE)
                else:
                    sub_model.setObjective(sum(gamma_val[i]*f[0, i] for i in range(num_buy))-sum(delta_val[i]*g[0, i] for i in range(num_sell)), GRB.MAXIMIZE)
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
        if 1e6-model.objVal >= ask:

            #get the index of all the non-zero in gamma and delta
            # gamma_index = np.where(gamma[0,:].x > 0)[0]
            # delta_index = np.where(delta[0,:].x > 0)[0]
            # print('current index: {}'.format(option_index))
            # print('sold to {} and bought from {}'.format(gamma_index, delta_index))
            opt_sell_book_frontier_labels[option_index] = 1

        else:
            print(f'finally original ask: {ask},quoted ask: {1e6-model.objVal}')
            #print the buy from and sell to with gamma and delta
            assert opt_sell_book_frontier_labels[option_index] == 0
        quote_price[option_df_index] = 1e6 - model.objVal
        for i in range(num_buy):
            if gamma[0,i].x > 0:
                print(f'buy {gamma[0,i].x} from {opt_buy_book[i,0]}({opt_buy_book[i,1]}+{opt_buy_book[i,2]},{opt_buy_book[i,3]}) at {opt_buy_book[i,-1]}')
        for i in range(num_sell):
            if delta[0,i].x > 0:
                print(f'sell {delta[0,i].x} to {opt_sell_book[i,0]}({opt_sell_book[i,1]}+{opt_sell_book[i,2]},{opt_sell_book[i,3]}) at {opt_sell_book[i,-1]}')

        # print(f'order to quote: {opt_sell_book_holder.iloc[option_index]}')
        # print(f'quote_price: {quote_price[option_df_index]}')
        print(f'\noriginal ask: {ask},\nquoted ask: {1e6-model.objVal}')
        breakpoint()

    # Fix the DataFrame creation by adding the new column name
    columns = list(opt_buy_book_holder.columns) + ['belongs_to_frontier']
    frontier_buy_book = pd.DataFrame(
        np.concatenate([opt_buy_book_holder, np.expand_dims(opt_buy_book_frontier_labels, axis=1)], axis=1),
        index=opt_buy_book_holder.index,
        columns=columns  # Use updated columns list
    )
    
    frontier_sell_book = pd.DataFrame(
        np.concatenate([opt_sell_book_holder, np.expand_dims(opt_sell_book_frontier_labels, axis=1)], axis=1),
        index=opt_sell_book_holder.index,
        columns=columns  # Use updated columns list
    )

    return pd.concat([frontier_buy_book, frontier_sell_book])





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


def run_with_timeout(opt_buy_book, opt_sell_book, opt_l = False):
    try:
        # 
        result = synthetic_combo_frontier_generation(opt_buy_book, opt_sell_book, opt_l = opt_l, debug=0)
        return result
    except GurobiError as e:
        print(f"Gurobi Error in process: {str(e)}")
        return None
    except Exception as e:
        print(f"Process error: {str(e)}")
        traceback.print_exc()
        return None
@contextmanager
def pool_context(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()

# Parameter lists
# NUM_STOCK_LIST = [2] #, 4, 8, 12, 16, 20]  # 12, 16, 20
# BOOK_SIZE_LIST = [50] #,150, 200, 250, 300, 350, 400]
# NOISE_LIST = [2**-6]#[2**-7, 2**-6, 2**-5, 2**-4, 2**-3]



if __name__ == '__main__':
    NUM_STOCK = args.num_stocks
    MARKET_SIZE = args.market_size
    NOISE = args.noise
    BOOK_SIZE = args.market_size
    SEED = args.seed
    WANDB_ENABLED = True
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        WANDB_ENABLED = False  # Disable wandb when saving to output_dir
    
    # Parse stock combination if provided
    stock_list = None
    if args.stock_combo:
        stock_list = args.stock_combo.split(',')
        print(f"Using specified stock combination: {stock_list}")
    
    # Create a new pool for each iteration
    tasks = {}
    directory_path = args.output_dir or f'/common/home/hg343/Research/accelerate_combo_option/data/combo_2_test'
    
    with pool_context(processes=20) as pool:
        try:
            # Initialize wandb only if enabled
            if WANDB_ENABLED:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"combo_frontier_num_stock_{NUM_STOCK}_noise_{NOISE}_market_size_{MARKET_SIZE}",
                )
            
            # Define stock selection based on arguments
            selection = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM']
            
            if not stock_list:
                # Use random selection if stock_combo not provided
                combinations = list(itertools.combinations(selection, NUM_STOCK))
                random_select_combination = list(random.choice(combinations))
                stock_list = random_select_combination
                stock_list  = ['BA', 'HD']
            combinations_string = '_'.join(stock_list)
            
            with wandb.init(
                project=args.wandb_project,
                name = f"combo_frontier_num_stock_{NUM_STOCK}_noise_{NOISE}_market_size_{MARKET_SIZE}",
            ) as run:
                #turn off random selection for now 
                selection = ['AAPL', 'AXP', 'BA', 'DIS', 'GS', 'HD', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MSFT', 'NKE', 'PG', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM']
                # combinations = list(itertools.combinations(selection,NUM_STOCK))
                #combination not seen ['MSFT', 'AXP'] , ['MSFT', 'GS'], ['JPM', 'IBM'], ['MSFT', 'BA'], ['NKE', 'IBM']]
                # combinations = [['MSFT', 'GS'] ]
                stock_list = ['GS', 'MSFT']
                combinations = [['GS', 'MSFT']]
                random_select_combination = list(random.choice(combinations))
                combinations_string = '_'.join(random_select_combination)
                for i in range(2,14):
                    filename = f'combinatorial/book/STOCK_2_SEED_{i}_book_{combinations_string}.npy'
                    if os.path.isfile(filename):
                        opt_book = np.load(filename)
                    else:
                        print('File not found')
                        opt_book, stock_list = gen_synthetic_combo_options(NUM_ST=NUM_STOCK, NUM_ORDER=args.num_orders, combinations= random_select_combination,SEED=i)
                        np.save(filename, opt_book)
                    num_books = len(opt_book)//50
                    # Create artifact once before the loop
                    artifact = wandb.Artifact(
                        name=f"combo_frontier_{'_'.join(stock_list)}_size_{BOOK_SIZE}_noise_{NOISE}",
                        type="dataset",
                        description="Collection of frontier options training data for different markets",
                        metadata={
                            'num_stock': NUM_STOCK,
                            'stock_name': '_'.join(stock_list),
                            'noise': NOISE,
                            'book_size': BOOK_SIZE,
                            'total_markets': num_books  # Add total number of markets
                        }
                    )
                    for market_index in tqdm(range(3,num_books), desc=f'Generating frontier for markets'):
                        stock_name = '_'.join(stock_list)
                        opt_book_1 = opt_book[market_index*MARKET_SIZE:(market_index+1)*MARKET_SIZE]
                        opt_buy_book, opt_sell_book = add_noise_orderbook(opt_book_1, NOISE)
                        print('#####Generating {} with size {} and noise {}#####'.format(filename, BOOK_SIZE, NOISE))
                                # Add debug print before async
                        print(f"Starting async computation for iteration {market_index}")
                        filename = f'combo_frontier_market_index_{market_index}_book_size_{BOOK_SIZE}_{stock_name}_NOISE_{NOISE}_offset_{args.offset}'
                        column_names = ['option1', 'option2', 'C=Call, P=Put','Strike Price of the Option Times 1000', 'transaction_type','B/A_price']
                        opt_orders_df = pd.DataFrame(np.concatenate([opt_buy_book, opt_sell_book], axis=0), columns=column_names)
                        # testing whether the generated frontiers themselves matches
                        # frontier_option_label_attempt_1 = synthetic_combo_frontier_generation(opt_buy_book, opt_sell_book, opt_l = args.offset, debug=0)
                        # frontier_option_label_attempt_1_copy = deepcopy(frontier_option_label_attempt_1)
                        # frontier_option_label_attempt_1.columns = [*column_names, 'belongs_to_frontier']
                        # opt_buy_book_new = frontier_option_label_attempt_1[frontier_option_label_attempt_1.loc[:, 'transaction_type']==1].iloc[:,:-1]
                        # opt_sell_book_new = frontier_option_label_attempt_1[frontier_option_label_attempt_1.loc[:, 'transaction_type']==0].iloc[:,:-1]
                        # frontier_option_label_attempt_2 = synthetic_combo_frontier_generation(opt_buy_book_new, opt_sell_book_new, opt_l = False, debug=0)
                        # assert frontier_option_label_attempt_2.equals(frontier_option_label_attempt_1_copy)
                        # frontier_option_label = deepcopy(frontier_option_label_attempt_1)
                        breakpoint()
                        result = synthetic_combo_frontier_generation(opt_orders_df, offset = args.offset)
                        print(result.iloc[:,4:])
                        breakpoint()
                        async_result = pool.apply_async(run_with_timeout, (opt_orders_df, args.offset))
                        # # frontier_option_label = async_result.get(timeout=600)
                        # df = pd.DataFrame(frontier_option_label, columns=['option1', 'option2','C=Call, P=Put',
                        # 'Strike Price of the Option Times 1000',
                        # 'transaction_type', 'B/A_price',
                        # 'belongs_to_frontier'])

                        # buy_book = df[df['transaction_type'] == 1]
                        # sell_book = df[df['transaction_type'] == 0]

                        # print(synthetic_combo_match_mip(buy_book, sell_book, debug=0)[3])
                        # with open('here.pkl', 'wb') as f:
                        #             pickle.dump(frontier_option_label, f)
                        # breakpoint()
                        breakpoint()
                        tasks.update({filename :[market_index, async_result]})
                    print(tasks)
                    for filename, solving_result in tasks.values():
                        market_index, async_result = solving_result
                        try:
                            print(f"Waiting for result in iteration {market_index}")
                            frontier_option_label = async_result.get(timeout=600)
                            print(frontier_option_label)
                            print(f"Result type for iteration {market_index}: {type(frontier_option_label)}")
                            if frontier_option_label is not None:
                                print(f"Successfully completed iteration {market_index}")
                                if not os.path.exists(directory_path):
                                    os.makedirs(directory_path)

                                stock_name = '_'.join(stock_list)
                                save_path = os.path.join(directory_path, filename+'.pkl')

                                print(f"Attempting to save to: {save_path}")
                                metadata = {
                                    'num_stock': NUM_STOCK,
                                    'stock_name': stock_name,
                                    'noise': NOISE,
                                    'book_size': BOOK_SIZE,
                                    'market_index': market_index
                                }
                                save_path = 'here.pkl'
                                with open(save_path, 'wb') as f:
                                    pickle.dump(frontier_option_label, f)
                                breakpoint()
                                # Log metadata for this specific market
                                wandb.log({
                                    f'market_{market_index}/metadata': metadata
                                })
                                
                                # Add this market's file to the artifact
                                artifact.add_file(save_path, name=f'markets/{filename}.pkl')
                                print(f"Successfully saved iteration {market_index}")
                            else:
                                print(f"Iteration {market_index} returned None")

                        except TimeoutError:
                            print(f"Iteration {market_index} timed out after 20 seconds")
                            # Pool will be automatically cleaned up when the context exits
                            continue

                # After the loop is complete, log the artifact once with all files
                run.log_artifact(artifact)

        except Exception as e:
            print(f"Error in main: {str(e)}")
            traceback.print_exc()
            raise e
