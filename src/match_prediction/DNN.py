import torch
import torch.nn as nn
import torch.optim as optim

# Define the DNN with two heads
class OptionOrderDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_stock_size=1, output_frontier_size=1):
        super(OptionOrderDNN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Head 1 for stock price prediction
        self.stock_head = nn.Linear(hidden_size, output_stock_size)
        
        # Head 2 for frontier set classification
        self.frontier_head = nn.Linear(hidden_size, output_frontier_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Shared layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        # Head 1 output (stock price)
        stock_price = self.stock_head(x)
        
        # Head 2 output (frontier set classification)
        frontier_prediction = self.sigmoid(self.frontier_head(x))
        
        return stock_price, frontier_prediction
    




