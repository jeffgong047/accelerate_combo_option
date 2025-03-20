import copy
from datetime import timedelta
from dateutil import parser
import pandas as pd
import yfinance as yf
import torch
from torch.nn.utils.rnn import pad_sequence
from gurobipy import *
import numpy as np 
import sys 
from tqdm import tqdm
from copy import deepcopy
from combinatorial.synthetic_combo_mip_match import synthetic_combo_match_mip
import timeit
from typing import List, Tuple, Dict, Optional, Union, Any, TYPE_CHECKING

# Reward functions for market matching
def profit_with_penalty_reward(buy_book, sell_book, reward_fn, full_df=None, penalty_weight=0.1):
    """
    Calculate reward as profit minus a penalty based on number of selected orders
    
    Args:
        buy_book: DataFrame of selected buy orders
        sell_book: DataFrame of selected sell orders
        reward_fn: Function to compute matching and profit
        full_df: Full DataFrame with all orders (for calculating penalty)
        penalty_weight: Weight for the selection penalty
        
    Returns:
        reward: Calculated reward
    """
    # Check if buy or sell book is empty
    if len(buy_book) == 0 or len(sell_book) == 0:
        return -1.0
    
    try:
        # Run matching algorithm - this should handle run_matching_with_timeout internally
        result = reward_fn(buy_book, sell_book)
        
        # Handle case where result is None (timeout or error occurred)
        if result is None:
            return -0.5
            
        # Unpack result based on the expected return format
        if isinstance(result, tuple) and len(result) >= 3:
            # For mechanism_solver_combo format: time, num_constraints, profit, isMatch, matched_stock
            if len(result) >= 5:
                _, _, profit, isMatch, _ = result
            # For mechanism_solver_single format: isMatch, profit
            elif len(result) == 2:
                isMatch, profit = result
            else:
                # Fallback for other formats
                profit = result[2]
                isMatch = profit > 0
        else:
            print(f"Unexpected result format: {result}")
            return -0.5
        
        # Calculate penalty based on number of selected orders
        if full_df is not None:
            num_selected = len(buy_book) + len(sell_book)
            max_orders = len(full_df)
            # Penalty increases with more selected orders
            selection_penalty = penalty_weight * (num_selected / max_orders)
            
            # Return profit minus penalty
            reward = profit - selection_penalty if isMatch else -0.5
        else:
            reward = profit if isMatch else -0.5
        
        return reward
        
    except Exception as e:
        print(f"Error in profit_with_penalty_reward: {e}")
        return -1.0

def profit_minus_liability_reward(buy_book, sell_book, reward_fn, liability_weight=0.1):
    """
    Calculate reward as profit minus a scaled sum of prices (liability)
    Based on mechanism_solver's liability constraints
    
    Args:
        buy_book: DataFrame of selected buy orders
        sell_book: DataFrame of selected sell orders
        reward_fn: Function to compute matching and profit
        liability_weight: Weight for scaling the liability penalty
        
    Returns:
        reward: Calculated reward
    """
    # Check if buy or sell book is empty
    if len(buy_book) == 0 or len(sell_book) == 0:
        return -1.0
    
    try:
        # Run matching algorithm - this should handle run_matching_with_timeout internally
        result = reward_fn(buy_book, sell_book)
        
        # Handle case where result is None (timeout or error occurred)
        if result is None:
            return -0.5
            
        # Unpack result based on the expected return format
        if isinstance(result, tuple) and len(result) >= 3:
            # For mechanism_solver_combo format: time, num_constraints, profit, isMatch, matched_stock
            if len(result) >= 5:
                _, _, profit, isMatch, _ = result
            # For mechanism_solver_single format: isMatch, profit
            elif len(result) == 2:
                isMatch, profit = result
            else:
                # Fallback for other formats
                profit = result[2]
                isMatch = profit > 0
        else:
            print(f"Unexpected result format: {result}")
            return -0.5
        
        # Only calculate liability if there's a match
        if isMatch:
            # Calculate liability as weighted exposure:
            # For buy orders: sum of (price * quantity)
            # For sell orders: sum of (price * quantity)
            # This mirrors how the mechanism solver models the constraint
            
            # Check if 'liquidity' column exists to use as quantity
            if 'liquidity' in buy_book.columns and 'liquidity' in sell_book.columns:
                # Buy liability (price * quantity for each buy order)
                buy_liability = (buy_book['B/A_price'] * buy_book['liquidity']).sum()
                # Sell liability (price * quantity for each sell order)
                sell_liability = (sell_book['B/A_price'] * sell_book['liquidity']).sum()
            else:
                # If no liquidity column, just use prices
                buy_liability = buy_book['B/A_price'].sum()
                sell_liability = sell_book['B/A_price'].sum()
            
            # Total liability is the sum of buy and sell liabilities
            total_liability = buy_liability + sell_liability
            
            # Return profit minus scaled liability (similar to the L term in mechanism_solver)
            reward = profit - liability_weight * total_liability
        else:
            reward = -0.5
        
        return reward
        
    except Exception as e:
        print(f"Error in profit_minus_liability_reward: {e}")
        return -1.0

def get_reward_function(reward_type='profit_with_penalty', **kwargs):
    """
    Factory function to get the appropriate reward function based on type
    
    Args:
        reward_type: Type of reward function to use
        **kwargs: Additional parameters for the reward function
        
    Returns:
        reward_function: The selected reward function
    """
    if reward_type == 'profit_minus_liability':
        liability_weight = kwargs.get('liability_weight', 0.1)
        return lambda buy_book, sell_book, reward_fn, full_df=None: profit_minus_liability_reward(
            buy_book, sell_book, reward_fn, liability_weight)
    else:  # Default to profit_with_penalty
        penalty_weight = kwargs.get('penalty_weight', 0.1)
        return lambda buy_book, sell_book, reward_fn, full_df=None: profit_with_penalty_reward(
            buy_book, sell_book, reward_fn, full_df, penalty_weight)

def collate_fn(batch):
    '''
    Collate function for the dataloader.
    Pads sequences of different lengths in a batch.
    
    Args:
        batch: List of (features, labels) tuples
        
    Returns:
        Tuple of (padded_features, padded_labels) tensors
    '''
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Convert to tensors if they're not already
    features = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in features]
    labels = [torch.tensor(y) if not isinstance(y, torch.Tensor) else y for y in labels]
    
    # Pad sequences
    padded_features = pad_sequence(features, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)
    
    return padded_features, padded_labels


        
