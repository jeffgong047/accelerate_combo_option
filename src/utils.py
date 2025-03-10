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


        
