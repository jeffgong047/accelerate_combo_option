import os
import numpy as np
import pandas as pd
import torch
from gurobipy import *
from match_prediction import BiAttentionClassifier, BiAttentionClassifier_single
from utils import Mechanism_solver_single, frontier_solver, Market
from copy import deepcopy
'''
implementation notes: 
market ship orders information 
strategy takes in order information
investor: has evaluator function that could compute(by taking in market information ) or load ground truth frontier 
'''

class MarketSimulator:
    def __init__(self, input_dir="../data/", price_date="20190123"):
        self.input_dir = input_dir
        self.price_date = price_date
        self.DJI = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
                    'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ',
                    'WBA', 'WMT', 'XOM']
        self.order_book = {}
        self.markets = []

    def load_market_data(self):
        """Load and preprocess market data 
            Load multiple many different markets. 
        """
        markets = [] 
        for stock in self.DJI[:1]:
            opt_stats_filename =os.path.join(self.input_dir, stock+"_"+self.price_date+".xlsx")
            opt_df_original = pd.read_excel(opt_stats_filename)
            expirations = sorted(list(set(opt_df_original['Expiration Date of the Option'])))
            for expiration in expirations[:1]:
                opt_df_period_market = opt_df_original[opt_df_original['Expiration Date of the Option'] == expiration]
                markets.append(Market(opt_df_period_market))
        self.markets = markets

    def present_markets(self):
        """Ship new orders to the market
        provide an iterable that one could get markets from. 
        The market data will be provided in the format of (unmatched orders, frontiers from unmatched orders,matched_orders)
        """
        # if stock not in self.order_book:
        #     self.order_book[stock] = []
        # self.order_book[stock].extend(orders)
        # return self.order_book[stock]
        return self.markets

class Exchange:
    def __init__(self, model, simulator):
        self.model = model
        self.simulator = simulator
        self.frontier_cache = {}
        self.results = {
            'predicted_profits': [],
            'true_profits': [],
            'execution_details': []
        }

    def prepare_frontier_data(self, market, frontier_labels= None,combo_option=False):
        """Load pre-computed ground truth frontier"""
        if combo_option is False and frontier_labels is None:
            is_match, profit ,_,matched_orders = Mechanism_solver_single(market)
            frontiers = frontier_solver(market)
            return is_match ,  profit, matched_orders, frontiers
        elif combo_option is False and frontier_labels is not None:
            is_match, profit ,_,matched_orders = Mechanism_solver_single(market)
            frontiers = frontier_labels.loc[frontier_labels['belongs_to_frontier'] == 1]
            return is_match ,  profit, matched_orders, frontiers
        elif combo_option is True and frontier_labels is not None:
            '''
            The market contains frontier 
            '''
            pass 




    def evaluate_profit(self, market_data):
        """Evaluate profit for a given stock
        Given the predicted frontier, true frontier, and market profit, evaluate the profit of the strategy
        """
        profit = 0
        for predicted_frontier, true_frontier, matched_orders, market_profit, market in market_data:
            # need to debug Mechanism_solver_single and ...
            NN_market = Market(pd.concat([predicted_frontier, matched_orders.iloc[:2]])) #predicted frontiers with few more orders
            true_market = Market(pd.concat([true_frontier, matched_orders.iloc[:2]])) #true frontiers with few more orders
            is_match, NN_frontier_profit,_, _ = Mechanism_solver_single(NN_market) #NN frontier profit
            is_match, true_frontier_profit, _,_ = Mechanism_solver_single(true_market) #true frontier profit
            strategy_profit = NN_frontier_profit - true_frontier_profit
            profit += strategy_profit
        return profit



def exchange_mechanism_approximation(orders, model):
    """
    Implement approximate the mechanism with DNN inference
    Args:
        orders: Current market orders (DataFrame with columns matching training data)
        model: Trained BiAttentionClassifier model
    Returns:
        DataFrame: Selected rows from original orders that form the frontier
    """
    # Create feature matrix with doubled size (bid and ask for each order)
    n_orders = len(orders)
    opt = np.zeros((2 * n_orders, 4), dtype=np.float32)
    
    # Keep track of original indices
    original_indices = []
    
    for i in range(n_orders):
        # First row: ask order
        opt[i] = [
            1 if orders.iloc[i]['C=Call, P=Put'] == 'C' else 0,
            orders.iloc[i]['Strike Price of the Option Times 1000'],
            orders.iloc[i]['Lowest  Closing Ask Across All Exchanges'],
            0  # transaction_type = 0 for ask
        ]
        original_indices.append(i)
        
        # Second row: bid order
        opt[i + n_orders] = [
            1 if orders.iloc[i]['C=Call, P=Put'] == 'C' else 0,
            orders.iloc[i]['Strike Price of the Option Times 1000'],
            orders.iloc[i]['Highest Closing Bid Across All Exchanges'],
            1  # transaction_type = 1 for bid
        ]
        original_indices.append(i)

    # Random shuffle the rows
    shuffle_indices = np.random.permutation(len(opt))
    opt = deepcopy(opt[shuffle_indices])
    original_indices = np.array(original_indices)[shuffle_indices]

    # Convert to tensor and get model predictions
    with torch.no_grad():
        model.eval()
        frontier_pred = model(torch.tensor(opt).unsqueeze(0))
        predicted_classes = frontier_pred.argmax(dim=-1)  # Get predicted classes (0 or 1)

    # Get original order indices that were selected as frontier
    selected_indices = original_indices[predicted_classes[0] == 1]
    
    # Return the selected rows from the original orders DataFrame
    return orders.iloc[selected_indices]



def main():
    #
    model_path = '/common/home/hg343/Research/accelerate_combo_option/models/frontier_option_classifier_single.pt'
    # Load trained model
    model = BiAttentionClassifier_single(input_size=4, hidden_size=32, num_classes=2, bidirectional=True)  # Model initialization
    model.load_state_dict(torch.load(model_path))  # Load the model state dictionary
    model.eval()  # Set the model to evaluation mode
    
    # Initialize simulator and investor
    simulator = MarketSimulator()
    simulator.load_market_data()
    markets = simulator.present_markets()
    exchange = Exchange(model, simulator)
    markets_profits = [] 
    markets_matched_orders = []
    markets_frontiers = []
    markets_predicted_frontiers = []
    num_markets = 1 
    for market in markets[:num_markets]:
        is_match ,  profit, matched_orders, frontiers  = exchange.prepare_frontier_data(market)
        predicted_frontiers = exchange_mechanism_approximation(market, model)
        # predicted_frontier, true_frontier, matched_indices, market_profit, market 
        markets_profits.append(profit)
        markets_matched_orders.append(matched_orders)
        markets_frontiers.append(frontiers)
        markets_predicted_frontiers.append(predicted_frontiers)
    # need to solve the view of the market data problem. The predicted frontier has different columns name compared to market data. 
    # Since linear program has assumed the input to be of particular format, we modify the trading strategy that gives market data format linear program assumes.
    #Market raw data preprocess should not be part of the linear program. It will be ambiguous to convert orders back to market raw data. 
    total_profit = exchange.evaluate_profit(zip(markets_predicted_frontiers, markets_frontiers, markets_matched_orders, markets_profits, markets))


    
    # Print results
    print(f"Total Profit: {total_profit}")

if __name__ == "__main__":
    main()
