import pdb
import sys
import os
import numpy as np
import pandas as pd

# example input: python3 single_stock_preprocess.py MSFT
input_dir = "../data/"
st = sys.argv[1]
price_date = "20190123"
expiration_date = 20190201
opt_stats_filename = os.path.join(input_dir, st+"_"+price_date+".xlsx")

opt_df = pd.read_excel(opt_stats_filename)
opt_df = opt_df[opt_df['Expiration Date of the Option']==expiration_date]
print('{} {} options expired on {}'.format(len(opt_df), st, expiration_date))
assert(np.array(opt_df['AM Settlement Flag'].astype(int)).all()==0), \
"options on the security expire at the market open of the last trading day."

# convert available options to matrix
# C/P, strike, bid, ask
opt = np.zeros((len(opt_df), 4))
for i in range(len(opt_df)):
	opt[i][0] = 1 if opt_df['C=Call, P=Put'].iloc[i] == 'C' else -1
	opt[i][1] = opt_df['Strike Price of the Option Times 1000'].iloc[i].astype(int)/1000
	opt[i][2] = opt_df['Highest Closing Bid Across All Exchanges'].iloc[i].astype(float)
	opt[i][3] = opt_df['Lowest  Closing Ask Across All Exchanges'].iloc[i].astype(float)
strikes = list(set(opt[:, 1]))
strikes.append(0)
strikes.append(sys.maxsize)

np.save(st+".npy", opt)
np.save(st+"_strikes.npy", sorted(strikes))