from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
from gurobipy import *
from copy import deepcopy
import pickle 
import random
import wandb 
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process stock options.')

    parser.add_argument('--output_file', type=str, default='single_frontier.pkl', help='Output file to save the training data (default: training_data_frontier_bid_ask.pkl)')
    parser.add_argument('--market_stream', type=str, default='../data/', help='Input directory to save the training data (default: ../data/)')
    parser.add_argument('--wandb_project', type=str, default='expediating_comb_financial_market_matching', help='Wandb project name (default: single_stock_frontier_data_preprocessor)')
    parser.add_argument('--offset', type = bool, default=False, help='Whether to allow offset for liability in the optimization')
    return parser.parse_args()

# Move the main execution code inside if __name__ == '__main__':
args = parse_arguments()

# Set wandb API key programmatically
os.environ["WANDB_API_KEY"] = "d1cb0d609d7b64218fe82a45a54e57f47e2d26da"

try:
    wandb.login()  # This will now use the API key we just set
except wandb.errors.AuthError:
    print("Could not authenticate with wandb. Invalid API key")
    sys.exit(1)

input_dir = args.market_stream
output_file = args.output_file
price_date = '20190123'
opt_l = args.offset
DJI = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', \
'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', \
'WBA', 'WMT', 'XOM']

# Initialize an empty list to hold the data
frontier_option_training_data_list = []   #frontier options, a list of pd dataframe that contains info about C/P, strike, bid, ask and belonging to frontier 
frontier_series = 0 


with wandb.init(
    project=args.wandb_project,
    name = f"{output_file}",
) as run:
    # Create metadata dictionary to store additional info
    metadata = {
        'price_date': price_date,
        'total_frontier_series': frontier_series,
        'num_stocks': len(DJI),
        'stocks_processed': DJI,
        'offset': opt_l,
        'data_generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    for st in DJI:
        opt_stats_filename = os.path.join(input_dir, st + "_" + price_date + ".xlsx")
        opt_df_original = pd.read_excel(opt_stats_filename)
        expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))
        
        for expiration_date in expirations:
            opt_df = opt_df_original[opt_df_original['Expiration Date of the Option'] == expiration_date]
            option_num = len(opt_df)
            
            # Convert available options to matrix C/P, strike, bid, ask
            # C/P ST STRIKE B/A strike price 
            opt = np.zeros((option_num, 6), dtype=object)
            for i in range(option_num):
                opt[i][0] = 1 if opt_df['C=Call, P=Put'].iloc[i] == 'C' else -1
                opt[i][1] = opt_df['Strike Price of the Option Times 1000'].iloc[i].astype(int) / 1000
                opt[i][2] = opt_df['Highest Closing Bid Across All Exchanges'].iloc[i].astype(float)
                opt[i][3] = opt_df['Lowest  Closing Ask Across All Exchanges'].iloc[i].astype(float)
                opt[i][4] = []  #storing whether the option belongs to frontier
                opt[i][5] = {'bid':[], 'ask':[]}  #storing the options that current option being dominated by
            strikes = list(set(opt[:, 1]))
            strikes.append(0)
            strikes.append(sys.maxsize)
            isMatch = 0 
            try:
                model = Model("match")
                model.setParam('OutputFlag', False)
                gamma = model.addVars(1, option_num, lb=0, ub=1) # sell to buys
                delta = model.addVars(1, option_num, lb=0, ub=1) # buy from asks
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
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num)) - l[0,0], GRB.MAXIMIZE)
                else:
                    model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num)), GRB.MAXIMIZE)
                model.optimize()
                # for v in model.getVars():
                # 	print('%s %g' % (v.varName, v.x))
                expense = 0
                for i in range(0, option_num):
                    if not (delta[0, i].x == 1 and gamma[0, i].x == 1):
                        if round(delta[0, i].x,4) != 0:
                            isMatch = 1
                            print('Buy {} {}({}, {}) at ask price {}'.format(round(delta[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', st, opt[i, 1], opt[i, 3]))
                            expense = expense - delta[0, i].x * opt[i, 3]
                        if round(gamma[0, i].x,4) != 0:
                            isMatch = 1
                            print('Sell {} {}({}, {}) at bid price {}'.format(round(gamma[0,i].x, 4), 'C' if opt[i, 0]==1 else 'P', st, opt[i, 1], opt[i, 2]))
                            expense = expense + gamma[0,i].x * opt[i, 2]
                # print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), round(l[0,0].x,2)))
                # print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), l))
            except GurobiError as e:
                print('Error code ' + str(e.errno) + ": " + str(e))
            except AttributeError:
                print('Encountered an attribute error')

            if not isMatch:
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
                        non_zero_values_gamma = {key[1]: var.X for key, var in gamma.items() if var.X != 0.0}
                        #sort the k,v based on v 
                        if len(non_zero_values_gamma) > 0:
                            values = non_zero_values_gamma.values()
                            keys = list(non_zero_values_gamma.keys())
                            sorted_keys_index = np.argsort(values)
                            for inx in sorted_keys_index:
                                opt[j][5]['bid'].append(keys[inx])   
                        else:
                            breakpoint() 
                    print('tighten_bid vesus bid: ', tighten_bid, bid )
                    # print('##########The highest bid for {}({}, {}) is {} (original {}). L is {}##########'.format(opt_type, st, opt[j, 1], round(tighten_bid,2), bid, round(l[0,0].x,2)))
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
                    else:
                        non_zero_values_delta = {key[1]: var.X for key, var in delta.items() if var.X != 0.0}
                        #sort the k,v based on v 
                        if len(non_zero_values_delta) > 0:
                            values = non_zero_values_delta.values()
                            keys = list(non_zero_values_delta.keys())
                            sorted_keys_index = np.argsort(values)
                            for inx in sorted_keys_index:
                                opt[j][5]['ask'].append(keys[inx])
                        else:
                            breakpoint()
                    # print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, round(l[0,0].x,2)))
                    # print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, l))
                    if tighten_bid == bid:
                        opt[j,4].append('bid')
                    if tighten_ask == ask:
                        opt[j,4].append('ask')
                    
                    
            #append frontier_option_training data for given stock and expiration date
            bid_frontier = pd.Series([1 if 'bid' in response else 0 for response in opt[:,4]])
            print(opt[:,5])
            bid_dominated_by = pd.Series([l.get('bid', []) for l in opt[:,5]])
            opt_bid_type = pd.Series(np.ones(len(bid_frontier)))
            opt_bid = pd.DataFrame(opt[:,[0,1,2]], columns = ['C=Call, P=Put','Strike Price of the Option Times 1000','B/A_price'])
            opt_bid['transaction_type'] = opt_bid_type
            opt_bid['belongs_to_frontier'] = bid_frontier
            opt_bid['dominated_by'] = bid_dominated_by
            ask_frontier = pd.Series([1 if 'ask' in response else 0 for response in opt[:,4]])
            ask_dominated_by = pd.Series([l.get('ask', []) for l in opt[:,5]])
            opt_ask_type = pd.Series(np.zeros(len(ask_frontier)))
            opt_ask = pd.DataFrame(opt[:,[0,1,3]], columns = ['C=Call, P=Put','Strike Price of the Option Times 1000','B/A_price'])
            opt_ask['transaction_type'] = opt_ask_type
            opt_ask['belongs_to_frontier'] = ask_frontier
            opt_ask['dominated_by'] = ask_dominated_by
            opt_bid = opt_bid.sample(frac=1).reset_index(drop=True)
            opt_ask = opt_ask.sample(frac=1).reset_index(drop=True)
            opt_frontier = pd.concat([opt_bid, opt_ask], axis=0)
            opt_frontier.attrs['price_date'] = opt_df_original['The Date of this Price'].iloc[0]
            opt_frontier.attrs['stock'] = st
            opt_frontier.attrs['expiration_date'] = expiration_date
            frontier_option_training_data_list.append(deepcopy(opt_frontier))


   # First save the data locally
    with open(output_file, 'wb') as f:
        pickle.dump(frontier_option_training_data_list, f)
    
    # Log the metadata
    wandb.log(metadata)
    
    # Save the pickle file to wandb
    artifact = wandb.Artifact(
        name=f"{output_file}",
        type="dataset",
        description="Frontier options training data",
        metadata=metadata
    )
    
    # Add the file to the artifact
    artifact.add_file(os.path.join(os.getcwd(), output_file))
    
    # Log the artifact to wandb
    run.log_artifact(artifact)
    
    print(f"Training data written to {output_file} and logged to wandb")
