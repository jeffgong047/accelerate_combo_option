import os
import numpy as np
import pandas as pd
from gurobipy import *
from copy import deepcopy
import pickle 
input_dir = "../data/"
output_file = "training_data.pkl"
price_date = "20190123"
opt_l = 1
DJI = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', \
'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', \
'WBA', 'WMT', 'XOM']

# Initialize an empty list to hold the data
frontier_option_training_data_list = []   #frontier options, a list of pd dataframe that contains info about C/P, strike, bid, ask and belonging to frontier 
frontier_series = 0 
for st in DJI:
    opt_stats_filename = os.path.join(input_dir, st + "_" + price_date + ".xlsx")
    opt_df_original = pd.read_excel(opt_stats_filename)
    expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))
    
    for expiration_date in expirations:
        opt_df = opt_df_original[opt_df_original['Expiration Date of the Option'] == expiration_date]
        option_num = len(opt_df)
        
        # Convert available options to matrix C/P, strike, bid, ask
        opt = np.zeros((option_num, 5))
        for i in range(option_num):
            opt[i][0] = 1 if opt_df['C=Call, P=Put'].iloc[i] == 'C' else -1
            opt[i][1] = opt_df['Strike Price of the Option Times 1000'].iloc[i].astype(int) / 1000
            opt[i][2] = opt_df['Highest Closing Bid Across All Exchanges'].iloc[i].astype(float)
            opt[i][3] = opt_df['Lowest  Closing Ask Across All Exchanges'].iloc[i].astype(float)
            opt[i][4] = None
        strikes = list(set(opt[:, 1]))
        strikes.append(0)
        strikes.append(sys.maxsize)

        for j in range(option_num):
            opt_type = 'C' if opt[j, 0]==1 else 'P'
            bid, ask = opt[j, 2], opt[j, 3]
            # finding the highest bid
            opt[j, 3] = 0 # exchange buys at ask
            try:
                model = Model("gen_match")
                model.setParam('OutputFlag', False)
                gamma = model.addVars(1, option_num, lb=0, ub=1)
                delta = model.addVars(1, option_num, lb=0, ub=1)
                # decision variable on the constraint
                if opt_l == 1:
                    l = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                for strike in sorted(strikes):
                    if opt_l == 1:
                        model.addLConstr(sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 2] > 0) - \
                            sum(delta[0, i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num)) - l[0,0], GRB.LESS_EQUAL, 0)
                    else:
                        model.addLConstr(sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 2] > 0) - \
                            sum(delta[0, i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num)), GRB.LESS_EQUAL, 0)
                if opt_l == 1:
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num))-l[0,0], GRB.MAXIMIZE)
                else:
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num)), GRB.MAXIMIZE)
                model.optimize()
                # for v in model.getVars():
                # 	print('%s %g' % (v.varName, v.x))
                for i in range(0, option_num):
                    if not (delta[0, i].x == 1 and gamma[0, i].x == 1):
                        if round(delta[0, i].x,4) != 0:
                            print('Buy {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', opt[i, 1], opt[i, 3]))
                        if round(gamma[0, i].x,4) != 0:
                            print('Sell {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', opt[i, 1], opt[i, 2]))
            except GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))
            except AttributeError:
                print('Encountered an attribute error')
            opt[j, 3] = ask
            tighten_bid = model.objVal
            if tighten_bid == bid:
                frontier_series += 1
            else:
                print('tighten_bid vesus bid: ', tighten_bid, bid )
            print('##########The highest bid for {}({}, {}) is {} (original {}). L is {}##########'.format(opt_type, st, opt[j, 1], round(tighten_bid,2), bid, round(l[0,0].x,2)))
            # print('##########The highest bid for {}({}, {}) is {} (original {}). L is {}##########'.format(opt_type, st, opt[j, 1], round(tighten_bid,2), bid, l))
            # finding the lowest ask
            opt[j, 2] = ask # exchange sells at bid
            try:
                model = Model("gen_match")
                model.setParam('OutputFlag', False)
                gamma = model.addVars(1, option_num, lb=0, ub=1)
                delta = model.addVars(1, option_num, lb=0, ub=1)
                # decision variable on the constraint
                if opt_l == 1:
                    l = model.addVars(1, 1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                for strike in sorted(strikes):
                    if opt_l == 1:
                        model.addLConstr(sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 2] > 0) - \
                            sum(delta[0, i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num)) - l[0,0], GRB.LESS_EQUAL, 0)
                    else:
                        model.addLConstr(sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 2] > 0) - \
                            sum(delta[0, i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num)), GRB.LESS_EQUAL, 0)
                if opt_l == 1:
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num))-l[0,0], GRB.MAXIMIZE)
                else:
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num)), GRB.MAXIMIZE)
                model.optimize()
                # for v in model.getVars():
                # 	print('%s %g' % (v.varName, v.x))
                for i in range(0, option_num):
                    if not (delta[0, i].x == 1 and gamma[0, i].x == 1):
                        if round(delta[0, i].x,4) != 0:
                            print('Buy {} {} option with strike {} at ask price {}'.format(round(delta[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', opt[i, 1], opt[i, 3]))
                        if round(gamma[0, i].x,4) != 0:
                            print('Sell {} {} option with strike {} at bid price {}'.format(round(gamma[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', opt[i, 1], opt[i, 2]))
            except GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))
            except AttributeError:
                print('Encountered an attribute error')
            opt[j, 2] = bid
            tighten_ask = ask - model.objVal
            if tighten_ask == ask:
                frontier_series += 1
            print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, round(l[0,0].x,2)))
            # print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, l))
            if tighten_bid == bid or tighten_ask == ask:
                opt[j,4] = 1
            else:
                opt[j,4] = 0 
        #append frontier_option_training data for given stock and expiration date
        frontier_option_training_data_point = pd.DataFrame(opt, columns = ['C=Call, P=Put','Strike Price of the Option Times 1000','Highest Closing Bid Across All Exchanges','Lowest  Closing Ask Across All Exchanges','belongs_to_frontier'])
        frontier_option_training_data_list.append(deepcopy(frontier_option_training_data_point))

with open(output_file, 'wb') as f:
    pickle.dump(frontier_option_training_data_list, f)


print(f"Training data written to {output_file}")
