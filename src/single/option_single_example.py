import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~48 DIS options expired on 20190621~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sell 1.0 C(DIS, 110.0) at bid price 7.2
# Buy 1.0 C(DIS, 150.0) at ask price 0.05
# Buy 1.0 P(DIS, 110.0) at ask price 5.1
# Sell 1.0 P(DIS, 150.0) at bid price 38.75
# The obj is 0.8: profit 40.8 now, subject to at most 40.0 loss in the future
# Arbitrage spotted for DIS option orders that expire on 20190621
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.linewidth'] = 2

dis = np.linspace(0, 200, 21)
buy_payoff = np.maximum(dis-150, 0) + np.maximum(110-dis, 0)
sell_payoff = np.maximum(dis-110, 0) + np.maximum(150-dis, 0)

plt.plot(dis, buy_payoff, 'black', linestyle='-', linewidth=3, label='buy portfolio')	
plt.plot(dis, sell_payoff, 'black', linestyle=':', linewidth=3, label='sell portfolio')
legend=plt.legend(frameon=False, fontsize='22', ncol=1, loc='upper right')
plt.setp(legend.get_title(),fontsize='20')
plt.axis([0-5, 200+5, 0-5, 150+5])
plt.xticks(fontsize=22)
plt.xticks(np.arange(0, 210, 40))
plt.yticks(fontsize=22)
plt.yticks(np.arange(0, 160, 50))
plt.xlabel('DIS', fontsize='22')
plt.ylabel('Payoff', fontsize='22')
plt.tick_params(length=5, width=1.5, direction="out", top=0, right=0)
plt.savefig("DIS_payoff.pdf", bbox_inches="tight")
plt.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~102 AAPL options expired on 20200117~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sell 1.0 C(AAPL, 160.0) at bid price 14.1
# Buy 1.0 C(AAPL, 80.0) at ask price 74.2
# Buy 1.0 P(AAPL, 160.0) at ask price 19.1
# Sell 1.0 P(AAPL, 80.0) at bid price 0.62
# The obj is 1.42: profit -78.58 now, subject to at most -80.0 loss in the future
# Arbitrage spotted for AAPL option orders that expire on 20200117
appl = np.linspace(0, 300, 31)
buy = np.maximum(appl-80, 0) + np.maximum(160-appl, 0)
sell = np.maximum(appl-160, 0) + np.maximum(80-appl, 0)

plt.plot(appl, buy, 'black', linestyle='-', linewidth=3, label='buy portfolio')	
plt.plot(appl, sell, 'black', linestyle=':', linewidth=3, label='sell portfolio')
legend=plt.legend(frameon=False, fontsize='22', ncol=1, loc='upper left')
plt.setp(legend.get_title(),fontsize='20')
plt.axis([0-5, 300+5, 0-5, 200+5])
plt.xticks(fontsize=22)
plt.xticks(np.arange(0, 310, 50))
plt.yticks(fontsize=22)
plt.yticks(np.arange(0, 210, 50))
plt.xlabel('AAPL', fontsize='22')
plt.ylabel('Payoff', fontsize='22')
plt.tick_params(length=5, width=1.5, direction="out", top=0, right=0)
plt.savefig("AAPL_payoff.pdf", bbox_inches="tight")
plt.close()