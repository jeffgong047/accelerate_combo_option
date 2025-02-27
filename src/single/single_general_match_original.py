import pdb
import sys
import os
import numpy as np
import pandas as pd
import datetime, time, dateutil
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gurobipy import *

input_dir = "../data/"
opt_l = 1
# opt_l = 0 # restrict l=0
# l = 0
price_date = "20190123"
DJI = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', \
'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', \
'WBA', 'WMT', 'XOM']
print(len(DJI))
max_profit, max_risk_free = 0, 0

for st in DJI:
	opt_stats_filename = os.path.join(input_dir, st+"_"+price_date+".xlsx")
	print(opt_stats_filename)
	opt_df_original = pd.read_excel(opt_stats_filename)
	expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))

	total_series, spread_series, frontier_series = 0, 0, 0
	call_spread, improved_call_spread = 0, 0
	put_spread, improved_put_spread = 0, 0
	L_positive, L_negative = [], []

	# iterate through expiration dates
	for expiration_date in expirations:
		opt_df = opt_df_original[opt_df_original['Expiration Date of the Option']==expiration_date]
		option_num = len(opt_df)
		total_series += option_num
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~{} {} options expired on {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(option_num, st, expiration_date))
		# convert available options to matrix C/P, strike, bid, ask
		opt = np.zeros((option_num, 4))
		for i in range(option_num):
			opt[i][0] = 1 if opt_df['C=Call, P=Put'].iloc[i] == 'C' else -1
			opt[i][1] = opt_df['Strike Price of the Option Times 1000'].iloc[i].astype(int)/1000
			opt[i][2] = opt_df['Highest Closing Bid Across All Exchanges'].iloc[i].astype(float)
			opt[i][3] = opt_df['Lowest  Closing Ask Across All Exchanges'].iloc[i].astype(float)
		strikes = list(set(opt[:, 1]))
		strikes.append(0)
		strikes.append(sys.maxsize)

		# match - scan for arbitrage first
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
			isMatch = 0
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
			print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), round(l[0,0].x,2)))
			# print('The obj is {}: profit {} now, subject to at most {} loss in the future'.format(round(model.objVal,2), round(expense,2), l))
		except GurobiError as e:
			print('Error code ' + str(e.errno) + ": " + str(e))
		except AttributeError:
			print('Encountered an attribute error')
		
		if isMatch:
			testing_hypothesis = True
		else:
			testing_hypothesis = False

		if isMatch:
			if l[0,0].x >= 0:
				L_positive.append([str(expiration_date), expense, l[0, 0].x, model.objVal])
			else:
				L_negative.append([str(expiration_date), expense, l[0, 0].x, model.objVal])
			# L_positive.append([str(expiration_date), model.objVal])
			print('Arbitrage spotted for {} option orders that expire on {}'.format(st, expiration_date))
		else:
			# tighten the spread
			frontiers_in_current_market = 0 
			spread_series += option_num
			spread, tighten_spread = 0, 0
			testing_hypothesis = isMatch

			for j in range(option_num):
				opt_type = 'C' if opt[j, 0]==1 else 'P'
				bid, ask = opt[j, 2], opt[j, 3]
				if opt_type == 'C':
					call_spread += ask - bid
				else:
					put_spread += ask - bid
				spread += ask - bid
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
					frontiers_in_current_market += 1
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
					frontiers_in_current_market += 1
				print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, round(l[0,0].x,2)))
				# print('##########The lowest ask for {}({}, {}) is {} (original {}). L is {}.##########'.format(opt_type, st, opt[j, 1], round(tighten_ask, 2), ask, l))
				if opt_type == 'C':
					improved_call_spread += tighten_ask - tighten_bid
				else:
					improved_put_spread += tighten_ask - tighten_bid
				tighten_spread += tighten_ask - tighten_bid

			if frontiers_in_current_market > 0 and testing_hypothesis: 
				print('is match but have frontiers')
				breakpoint()
				a=1
				b=2
				c=a+b
			elif frontiers_in_current_market == 0 and not testing_hypothesis:
				print('is not match but no frontiers')
				breakpoint()
			print('The total spread is improved by 1-{}/{} = {}'.format(round(tighten_spread,2), round(spread,2), round(1-tighten_spread/spread, 2)))

	# # results across all expirations	
	p_list = []
	if L_positive:
		print(L_positive)
		for element in L_positive:
			p_list.append(element[1]) #3
		print(p_list)
		max_profit = max(max_profit, max(p_list))
	date = datetime.datetime(year=int(price_date[0:4]), month=int(price_date[4:6]), day=int(price_date[6:8]))
	r_list = []
	# L_negative.append([str(expiration_date), expense, l[0, 0].x, model.objVal])
	if L_negative:
		print(L_negative)
		for element in L_negative:
			exp_date = element[0]
			diff_day = datetime.datetime(year=int(exp_date[0:4]), month=int(exp_date[4:6]), day=int(exp_date[6:8])) - date
			diff_year = diff_day.days/365
			r = np.log(np.abs(element[2])/np.abs(element[1]))/diff_year
			r_list.append(r)
		print(r_list)
		max_risk_free = max(max_risk_free, max(r_list))

	ave_call = call_spread/(spread_series/2)
	ave_put = put_spread/(spread_series/2)
	imp_call = improved_call_spread/(spread_series/2)
	imp_put = improved_put_spread/(spread_series/2)

	# print('total #markets = {}, #exp = {}, ave #markets = {}'.format(\
	# 	total_series, len(expirations), round(total_series/len(expirations), 2)))
	print('len_lp = {}, ave_p = {}, len_ln = {}, ave_r = {}'.format(\
		len(L_positive), 0 if not p_list else np.mean(p_list), len(L_negative), 0 if not r_list else np.mean(r_list)))
	print('len_lp = {}, ave_p = {}'.format(\
		len(L_positive), 0 if not p_list else np.mean(p_list)))
	print('#spread markets = {}, #frontier markets = {}'.format(\
		spread_series, frontier_series))
	print('ave call = {}, ave put = {}, ave imp_call = {}, ave imp_put = {}, imp perc = {}'.format(\
		round(ave_call,2), round(ave_put,2), round(imp_call,2), round(imp_put,2), round(1-(imp_call+imp_put)/(ave_call+ave_put),2)))

print(max_profit)
print(max_risk_free)