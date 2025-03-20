import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super().__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def attention_net(self, x):
        # Self-attention mechanism
        attn_scores = torch.bmm(x, x.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.bmm(attn_weights, x)
        return attended

    def forward(self, x):
        # Attention block with residual connection and layer norm
        attended = self.attention_net(x)
        x = self.layer_norm1(x + attended)
        
        # FFN block with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class BiAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3, dropout_rate=0.2):
        super().__init__()
        
        # Initial projection layer
        self.linear = nn.Linear(input_size, hidden_size)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Final classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Policy head (initialized as None)
        self.policy_head = None

    def init_policy_head(self):
        """Initialize the policy head for fine-tuning"""
        # Initialize policy head with same output dimension as main classifier
        hidden_size = self.fc.in_features
        num_classes = self.fc.out_features
        self.policy_head = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights (optional but recommended)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def get_embedding(self, x):
        return self.linear(x)

    def check_attention_score(self, x):
        x = self.linear(x)
        # Return attention weights from the first transformer block
        attn_scores = torch.bmm(x, x.transpose(1, 2))
        return F.softmax(attn_scores, dim=-1)

    def forward(self, x, use_policy_head=False):
        # Initial projection
        x = self.linear(x)
        
        # Pass through transformer blocks
        if not use_policy_head:
            with torch.no_grad():
                for block in self.transformer_blocks:
                    x = block(x)
        else:
            for block in self.transformer_blocks:
                x = block(x)
        
        # Use policy head if specified and initialized
        if use_policy_head and self.policy_head is not None:
            return self.policy_head(x)
        
        # Otherwise use the main classifier
        return self.fc(x)

class DQNWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, base_model=None):
        super(DQNWithEmbeddings, self).__init__()
        
        # Base model to extract embeddings
        self.base_model = base_model
        
        # Determine input dimension based on whether we're using a base model
        self.input_dim = hidden_size if base_model else input_size
        
        # Q-Network layers
        self.q_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
    def forward(self, x, state=None):
        """
        Forward pass through the network
        Args:
            x: Input tensor representing market data
            state: Current state represented as a binary vector (0/1 for each order)
                   If None, will be derived from market data
        """
        # Get embeddings from base model if available
        if self.base_model:
            with torch.no_grad():
                embeddings = self.base_model.get_embedding(x)
            
            # If we have a state vector, concatenate with embeddings
            if state is not None:
                # Reshape state to match embeddings
                state_tensor = state.unsqueeze(-1).expand_as(embeddings)
                # Combine state information with embeddings
                combined = torch.cat([embeddings, state_tensor], dim=-1)
                return self.q_network(combined)
            else:
                return self.q_network(embeddings)
        else:
            # Use input directly if no base model
            return self.q_network(x)




