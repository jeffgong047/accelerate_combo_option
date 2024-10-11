import pdb
import numpy as np
from synthetic_combo_mip_match import *
from gurobipy import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

opt_book = []
opt_book.append([1, 2, 1, 300, 1, 110]) #sell
opt_book.append([1, 1, 1, 300, 1, 70])
opt_book.append([1, 3, 1, 300, 0, 160]) #buy
opt_book.append([1, 0, 1, 250, 0, 5])
opt_book = np.array(opt_book)
opt_buy_book, opt_sell_book = opt_book[opt_book[:, -2]==1], opt_book[opt_book[:, -2]==0]

print(opt_buy_book)
print(opt_sell_book)

synthetic_combo_match_mip(opt_buy_book, opt_sell_book, s1='AAPL', s2='MSFT', debug=1)

# Sell 1.0 to C(1AAPL+2MSFT,300) at bid price 110
# Sell 1.0 to C(1AAPL+1MSFT,300) at bid price 70
# Buy 1.0 from C(1AAPL+3MSFT,300) at ask price 160
# Buy 1.0 from C(1AAPL+0MSFT,250) at ask price 5
# Revenue at T0 is 15.0; L is 0.0; Objective is 15.0 = 15.0

#plot
def buy_payoff(a, m):
	return np.maximum(a+3*m-300, 0) + np.maximum(a-250, 0)
def sell_payoff(a, m):
	return np.maximum(a+2*m-300, 0) + np.maximum(a+m-300, 0)
def payoff(a, m):
	return np.maximum(a+3*m-300, 0) + np.maximum(a-250, 0) - \
	np.maximum(a+2*m-300, 0) - np.maximum(a+m-300, 0)
a = np.linspace(0, 500, 11)
m = np.linspace(0, 500, 11)
A, M = np.meshgrid(a, m)
pay = payoff(A, M)
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.zaxis.set_tick_params(labelsize=16)
ax.tick_params(direction='out', length=8, width=6)

p = ax.plot_wireframe(A, M, pay, color='black')
ax.set_xlabel('AAPL', fontsize='20', labelpad=15)
ax.set_ylabel('MSFT', fontsize='20', labelpad=20)
ax.set_zlabel('Payoff', fontsize='20', labelpad=10)
fig.tight_layout()
#plt.show()
plt.savefig("combo_example.pdf", bbox_inches="tight")