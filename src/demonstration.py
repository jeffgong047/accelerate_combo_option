import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from match_prediction import BiAttentionClassifier,train_model, validate_model, test_model
import numpy as np
import pickle
from torch.nn.utils.rnn import pad_sequence
from utils import collate_fn
from tqdm import tqdm
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Option Order DNN with custom hyperparameters")
    parser.add_argument('--input_size', type=int, default=4, help='Number of input features (e.g. bid/ask prices)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test set size as a percentage of the entire data')
    return parser.parse_args()


def main():
    args = parse_arguments()
    sanity_check = False
    # Assuming your data is loaded and preprocessed here
    # X = bid_ask_prices_features, y = frontier_targets
    # Split the data into training, validation, and test sets
    # data = pd.read_csv('training_data.csv')
    with open('training_data.pkl','rb') as f:
        data = pickle.load(f)
    X = [x[['C=Call, P=Put','Strike Price of the Option Times 1000','Highest Closing Bid Across All Exchanges','Lowest  Closing Ask Across All Exchanges']].to_numpy(dtype = np.float32) for x in data]
    y = [y['belongs_to_frontier'].to_numpy(dtype = np.float32) for y in data]
    attention_mechanism_distribution = []
    #load model
    data_loader = DataLoader(list(zip(X, y)), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = BiAttentionClassifier(input_size=args.input_size, hidden_size=args.hidden_size,  num_classes=2, bidirectional=True)
    model_state_dict = torch.load(f'best_frontier_option_classifier_attention.pt')
    model.load_state_dict(model_state_dict)
    model.eval()
    attention_proportions = []
    attention_stock_price_ratios =[]
    for batch in tqdm(data_loader):
        sample_checking_raw, sample_target_raw  = batch
        if sanity_check:
            sample_checking = torch.cat([sample_checking_raw, sample_checking_raw], dim=1)
            sample_target = torch.cat([sample_target_raw, sample_target_raw], dim=1)
        else:
            sample_checking = sample_checking_raw
            sample_target = sample_target_raw

        with torch.no_grad():
            attention_weights = model.check_attention_score(sample_checking).squeeze(0)
            breakpoint()

        sorted_attention_weights , sorted_index = torch.sort(attention_weights, descending=True)
        # given the sorted_index, we want to test on Hypothesis:
        # All instances whether, the highest attention, 95% of attention is located on two options which
        #stock price estimation falls on two extreme end of collective stock price estimation.

        average_attention_options = torch.sum(attention_weights, dim = 0)/attention_weights.shape[0]
        average_attention_option_sorted, average_attention_options_sorted_index = torch.sort(average_attention_options)
        attention0_option_idx, attention1_option_idx = average_attention_options_sorted_index[0], average_attention_option_sorted_index[1]
        attention_proportion =  (average_attention_option_sorted[0] + average_attention_option_sorted[1])/ torch.sum(average_attention_options)
        attention_proportions.append(attention_proportion)
        # sample_target = sample_target.unsqueeze(0)
        # sample_target = sample_target.expand(-1,sample_target.shape[-1], -1)
        # sample_target_squeezed = sample_target.squeeze(0)  # Shape: [114, 114]
        # sorted_index_squeezed = sorted_index.squeeze(0)    # Shape: [114, 114]
        #
        # # Use torch.gather to reorder sample_target_squeezed based on sorted_index along the last dimension
        # sample_target_sorted = sample_target_squeezed.gather(1, sorted_index_squeezed)
        #
        # #sanity check.. top 50 percent attention weight  versus bottom 50 percent attention weight on the target
        # top_50_percent_labels = torch.sum(sample_target_sorted[:,:sample_target_sorted.shape[-1]//2])
        # bottom_50_percent_labels = torch.sum(sample_target_sorted[:, sample_target_sorted.shape[-1]//2:])
        # attention_distribution_sample = [top_50_percent_labels, bottom_50_percent_labels]
        # print(top_50_percent_labels/(top_50_percent_labels+bottom_50_percent_labels),top_50_percent_labels+bottom_50_percent_labels)
        #
        # attention_mechanism_distribution.append(attention_distribution_sample)
        # # Convert tensors to numpy for plotting
        # sorted_attention_weights_np = sorted_attention_weights.cpu().numpy().squeeze(0)
        # sorted_sample_target_np = sample_target_sorted.cpu().numpy()
        # Prepare annotations with symbols for label=1
        #calculate the estimated stock price
        stock_price_estimation = ((sample_checking[0,:,2] + sample_checking[0,:,3])/2*sample_checking[0,:,0] + sample_checking[0,:,1])
        stock_price_estimation_range = torch.max(stock_price_estimation) - torch.min(stock_price_estimation)
        attention_stock_price_range = torch.abs(stock_price_estimation[attention0_option_idx], stock_price_estimation[attention1_option_idx])
        attention_stock_price_ratio = attention_stock_price_range/stock_price_estimation_range
        attention_stock_price_ratios.append(attention_stock_price_ratio)
        average_stock_estimation = torch.sum(stock_price_estimation)/sample_checking.shape[1]
        abs_diff_stock_price_estimation = torch.abs(stock_price_estimation - average_stock_estimation)
        abs_diff_sorted , option_index_abs_diff_sorted = torch.sort(abs_diff_stock_price_estimation)
        print(option_index_abs_diff_sorted)
        attention_weights_np = attention_weights.cpu().numpy().squeeze(0)
        sample_target_np  = np.tile(sample_target.cpu().numpy(),(attention_weights_np.shape[0],1))

        print(attention_weights_np.shape)
        print(sample_target_np.shape)
        annotations = np.empty_like( sample_target_np, dtype=str)
        annotations[sample_target_np == 1] = '*'  # Mark label=1 with '*'
        annotations[sample_target_np == 0] = ''   # Leave label=0 cells blank
        # Plot the heatmap with conditional annotations
        # if sanity_check:
        #     plt.figure(figsize=(16, 12))
        # else:
        #     plt.figure(figsize=(8,6))
        # ax = sns.heatmap(np.log1p(attention_weights_np), annot=annotations, fmt="",
        #                  cmap="viridis", cbar=True, square=True, annot_kws={"size": 10})
        #
        # # Set plot title and labels
        # halfway = attention_weights.shape[1] // 2
        # if sanity_check:
        #     plt.axvline(x=halfway, color='red', linestyle='-', linewidth=5)
        # # ax.set_title(f"Attention Weights with Target Labels: high attention on frontier percentile (top 50 vs bottom 50) = {top_50_percent_labels/(top_50_percent_labels+bottom_50_percent_labels)}")
        # ax.set_xlabel("option index")
        # ax.set_ylabel("option index")
        # if sanity_check:
        #     plt.savefig('sanity_check_attention_map')
        # else:
        #     plt.savefig(f'attention_map')
        # breakpoint()
    print(torch.av)
    breakpoint()
    attention_mechanism_distribution_np = np.log1p(np.array(attention_mechanism_distribution))
    np.savetxt('attention_mechanism_distribution.txt',attention_mechanism_distribution_np)
    top_50_sum = np.sum(attention_mechanism_distribution_np[:, 0])
    total_sum = np.sum(attention_mechanism_distribution_np)
    proportion_top_50 = top_50_sum / total_sum
    print("Proportion of top 50% labels to total labels:", proportion_top_50)




if __name__ == '__main__':
    main()