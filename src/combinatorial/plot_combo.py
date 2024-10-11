import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pdb

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.linewidth'] = 2

option = sys.argv[1]
# STOCK = 4; NOISE = 2^-4
if option == 'book':
	book_size_result = np.load('results/combo_vary_book_size.npy')
	num_sim = book_size_result.shape[1]
	book_size_mean = np.mean(book_size_result, 1)
	book_size_se = np.std(book_size_result, 1)/np.sqrt(num_sim)
	book_size = [50, 100, 150, 200, 250, 300, 350, 400]
	num_constr = book_size_mean[:, 0]
	constr_err = book_size_se[:, 0]
	rev = book_size_mean[:, 1]
	rev_err = book_size_se[:, 1]

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('#Orders', fontsize=22)
	ax1.set_ylabel('#Iterations', fontsize=22)
	ax1.set_xlim(50-20, 400+20)
	ax1.set_xticks([50, 100, 200, 300, 400])
	ax1.set_ylim(0-5, 300+15)
	ax1.set_yticks([0, 100, 200, 300])
	ax1.tick_params(axis='x', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.tick_params(axis='y', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.errorbar(book_size, num_constr, constr_err, marker='D', linestyle='-', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='iterations')

	ax2 = ax1.twinx()
	ax2.set_ylabel('Net Profit', fontsize=22)
	ax2.set_ylim(0-5, 320+15)
	ax2.set_yticks([0, 80, 160, 240, 320]) #np.arange(0, 260, 50)
	ax2.tick_params(axis='y', length=5, width=1.5, direction="out", \
		labelcolor='black', labelsize=22)
	ax2.errorbar(book_size, rev, rev_err, marker='D', linestyle='--', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='profit')
	
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	ax1.legend(lines, labels, frameon=False, fontsize='18', ncol=1, loc="lower right")

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig("results/combo_vary_book_size.pdf", bbox_inches="tight")
	plt.close()

# NOISE = 2^-4, BOOK_SIZE = 150
if option == 'stock':
	stock_size_result = np.load('results/combo_vary_stock_size.npy')
	stock_size_constrained_result = np.load('results/combo_vary_stock_size_constrained_pairs.npy')
	num_sim = stock_size_result.shape[1]
	stock_size_mean = np.mean(stock_size_result, 1)
	stock_size_se = np.std(stock_size_result, 1)/np.sqrt(num_sim)
	stock_size_constr_mean = np.mean(stock_size_constrained_result, 1)
	stock_size_constr_se = np.std(stock_size_constrained_result, 1)/np.sqrt(num_sim)
	stock_size = [2, 4, 8, 12, 16, 20]

	num_constr = stock_size_mean[:, 0]
	constr_err = stock_size_se[:, 0]
	rev = stock_size_mean[:, 1]
	rev_err = stock_size_se[:, 1]

	num_constr_ = [num_constr[0]] + list(stock_size_constr_mean[:, 0])
	constr_err_ = [constr_err[0]] + list(stock_size_constr_se[:, 0])
	rev_ = [rev[0]] + list(stock_size_constr_mean[:, 1])
	rev_err_ = [rev_err[0]] + list(stock_size_constr_se[:, 1])

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('#Stocks (U)', fontsize=22)
	ax1.set_ylabel('#Iterations', fontsize=22)
	ax1.set_xlim(0-0.5, 20+0.5)
	ax1.set_xticks(np.arange(0, 22, 4))
	ax1.set_ylim(0-30, 1500+30)
	ax1.set_yticks([0, 500, 1000, 1500])
	ax1.tick_params(axis='x', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.tick_params(axis='y', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.errorbar(stock_size, num_constr, constr_err, marker='D', linestyle='-', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='iterations')
	ax1.errorbar(stock_size, num_constr_, constr_err_, marker='D', linestyle='-', \
		linewidth=3, elinewidth=2, color='red', capsize=3, capthick=2) #, label='iterations (restricted pairs)')

	ax2 = ax1.twinx()
	ax2.set_ylabel('Net Profit', fontsize=22)
	ax2.set_ylim(0-5, 200+5)
	ax2.set_yticks([0, 50, 100, 150, 200]) #np.arange(0, 260, 50)
	ax2.tick_params(axis='y', length=5, width=1.5, direction="out", \
		labelcolor='black', labelsize=22)
	ax2.errorbar(stock_size, rev, rev_err, marker='D', linestyle='--', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='profit')
	ax2.errorbar(stock_size, rev_, rev_err_, marker='D', linestyle='--', \
		linewidth=3, elinewidth=2, color='red', capsize=3, capthick=2) #, label='profit (restricted pairs)')
	
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	ax1.legend(lines, labels, frameon=False, fontsize='18', ncol=1, loc="lower right")

	fig.tight_layout()
	plt.savefig("results/combo_vary_stock_size.pdf", bbox_inches="tight")
	plt.close()

# STOCK = 4; BOOK_SIZE = 150
if option == 'noise':
	noise_result = np.load('results/combo_vary_noise_level.npy')
	num_sim = noise_result.shape[1]
	noise_mean = np.mean(noise_result, 1)
	noise_se = np.std(noise_result, 1)/np.sqrt(num_sim)
	noise_level = [-7, -6, -5, -4, -3]

	num_constr = noise_mean[:, 0]
	constr_err = noise_se[:, 0]
	rev = noise_mean[:, 1]
	rev_err = noise_se[:, 1]

	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Noise Level ($\log_2(\eta)$)', fontsize=22)
	ax1.set_ylabel('#Iterations', fontsize=22)
	ax1.set_xlim(-7-0.1, -3+0.1)
	ax1.set_xticks(noise_level)
	ax1.set_ylim(0-5, 300+5)
	ax1.set_yticks([0, 100, 200, 300])
	ax1.tick_params(axis='x', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.tick_params(axis='y', length=5, width=1.5, direction="out", \
		top=0, right=0, labelcolor='black', labelsize=22)
	ax1.errorbar(noise_level, num_constr, constr_err, marker='D', linestyle='-', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='iterations')

	ax2 = ax1.twinx()
	ax2.set_ylabel('Net Profit', fontsize=22)
	ax2.set_ylim(0-5, 250+5)
	ax2.set_yticks([0, 50, 100, 150, 200, 250]) #np.arange(0, 260, 50)
	ax2.tick_params(axis='y', length=5, width=1.5, direction="out", \
		labelcolor='black', labelsize=22)
	ax2.errorbar(noise_level, rev, rev_err, marker='D', linestyle='--', \
		linewidth=3, elinewidth=2, color='black', capsize=3, capthick=2, label='profit')
	
	lines_1, labels_1 = ax1.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	lines = lines_1 + lines_2
	labels = labels_1 + labels_2
	ax1.legend(lines, labels, frameon=False, fontsize='18', ncol=1, loc="lower right")

	fig.tight_layout()
	plt.savefig("results/combo_vary_noise_level.pdf", bbox_inches="tight")
	plt.close()