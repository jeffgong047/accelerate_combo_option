import os
import numpy as np
import pandas as pd
from gurobipy import *
from utils import get_stock_price

input_dir = "../data/"
output_file = "training_data.csv"
price_date = "20190123"
DJI = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', \
'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', \
'WBA', 'WMT', 'XOM']

# Initialize an empty list to hold the data
training_data = []

for st in DJI:
    opt_stats_filename = os.path.join(input_dir, st + "_" + price_date + ".xlsx")
    opt_df_original = pd.read_excel(opt_stats_filename)
    expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))
    
    for expiration_date in expirations:
        st_exp = get_stock_price(stock=st, date=expiration_date)
        opt_df = opt_df_original[opt_df_original['Expiration Date of the Option'] == expiration_date]
        option_num = len(opt_df)
        
        # Convert available options to matrix C/P, strike, bid, ask
        opt = np.zeros((option_num, 4))
        for i in range(option_num):
            opt[i][0] = 1 if opt_df['C=Call, P=Put'].iloc[i] == 'C' else -1
            opt[i][1] = opt_df['Strike Price of the Option Times 1000'].iloc[i].astype(int) / 1000
            opt[i][2] = opt_df['Highest Closing Bid Across All Exchanges'].iloc[i].astype(float)
            opt[i][3] = opt_df['Lowest  Closing Ask Across All Exchanges'].iloc[i].astype(float)
        
        strikes = list(set(opt[:, 1]))
        strikes.append(0)
        strikes.append(sys.maxsize)

        # Run the optimization to classify options into frontier set
        try:
            model = Model("match")
            model.setParam('OutputFlag', False)
            gamma = model.addVars(1, option_num, lb=0, ub=1)  # sell to buys
            delta = model.addVars(1, option_num, lb=0, ub=1)  # buy from asks

            for strike in sorted(strikes):
                model.addLConstr(sum(gamma[0,i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num) if opt[i, 2] > 0) - \
                                 sum(delta[0, i]*max(opt[i, 0]*(strike-opt[i, 1]), 0) for i in range(option_num)), GRB.LESS_EQUAL, 0)
            model.setObjective(sum(gamma[0,i]*opt[i, 2] for i in range(option_num)) - sum(delta[0,i]*opt[i, 3] for i in range(option_num)), GRB.MAXIMIZE)
            model.optimize()

            # Label data based on optimization results
            for i in range(option_num):
                expiration = expiration_date
                strike_price = opt[i, 1]
                order_type = 'Bid' if opt[i, 2] > 0 else 'Ask'
                value = opt[i, 2] if opt[i, 2] > 0 else opt[i, 3]
                
                # Determine if the option is in the frontier set based on Gurobi results
                is_frontier = (round(delta[0, i].x, 4) != 0 or round(gamma[0, i].x, 4) != 0)
                
                # Append the feature and label to the training_data list
                training_data.append([st, expiration, strike_price, order_type, value, int(is_frontier)])
        
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
        except AttributeError:
            print('Encountered an attribute error')

# Convert the data to a DataFrame
df = pd.DataFrame(training_data, columns=['Stock', 'Expiration Date', 'Strike Price', 'Order Type', 'Value', 'Frontier Label'])

# Write the data to CSV file
df.to_csv(output_file, index=False)

print(f"Training data written to {output_file}")
