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
        
        # For the combined network, we'll use a separate embedding for the state
        # and then combine it with the order embeddings
        self.state_embedding = nn.Linear(1, self.input_dim)
        
        # Attention mechanism to combine order embeddings with state
        self.attention = nn.Linear(self.input_dim * 2, 1)
        
        # Q-Network layers - takes the combined embeddings after attention
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
            x: Input tensor representing market data [batch_size, num_orders, features]
            state: Current state represented as a binary vector (0/1 for each order)
                   [batch_size, num_orders] or [num_orders]
        """
        batch_size = x.size(0)
        
        # Get embeddings from base model if available
        if self.base_model:
            # Get embeddings [batch_size, num_orders, hidden_size]
            embeddings = self.base_model.get_embedding(x)
            
            # If we have a state vector, use attention to combine with embeddings
            if state is not None:
                # Process state to proper format
                if state.dim() == 1:
                    # Single state vector [num_orders] -> [1, num_orders]
                    state = state.unsqueeze(0).float()
                else:
                    # Batch of state vectors [batch_size, num_orders]
                    state = state.float()
                
                # Make sure batch size matches
                if state.size(0) != batch_size:
                    state = state.repeat(batch_size, 1)
                
                # If state has more/fewer orders than embeddings, pad/truncate
                num_orders_emb = embeddings.size(1)
                num_orders_state = state.size(1)
                
                if num_orders_state > num_orders_emb:
                    # Truncate state
                    state = state[:, :num_orders_emb]
                elif num_orders_state < num_orders_emb:
                    # Pad state with zeros
                    padding = torch.zeros(batch_size, num_orders_emb - num_orders_state, 
                                         device=state.device)
                    state = torch.cat([state, padding], dim=1)
                
                # Create state embeddings [batch_size, num_orders, hidden_size]
                state = state.unsqueeze(-1)  # [batch_size, num_orders, 1]
                state_embeddings = self.state_embedding(state)  # [batch_size, num_orders, hidden_size]
                
                # Concatenate order embeddings and state embeddings
                # [batch_size, num_orders, hidden_size*2]
                combined = torch.cat([embeddings, state_embeddings], dim=-1)
                
                # Compute attention scores [batch_size, num_orders, 1]
                attention_scores = self.attention(combined)
                attention_weights = F.softmax(attention_scores, dim=1)
                
                # Apply attention to get weighted embeddings
                # [batch_size, num_orders, hidden_size] * [batch_size, num_orders, 1]
                # -> [batch_size, num_orders, hidden_size]
                weighted_embeddings = embeddings * attention_weights
                
                # Sum over orders to get a single vector per batch
                # [batch_size, hidden_size]
                context_vectors = weighted_embeddings.sum(dim=1)
                
                # Process through Q-network
                # [batch_size, num_actions]
                q_values = self.q_network(context_vectors)
                
                return q_values
            else:
                # Without state, just average the embeddings
                # [batch_size, hidden_size]
                context_vectors = embeddings.mean(dim=1)
                
                # Process through Q-network
                # [batch_size, num_actions]
                q_values = self.q_network(context_vectors)
                
                return q_values
        else:
            # Without base model, we expect x to already be a feature vector
            # [batch_size, features]
            if state is not None:
                # For simplicity, just concatenate with state and process
                # [batch_size, features + state_size]
                # We need to ensure state is properly sized
                if state.dim() == 1:
                    state = state.unsqueeze(0).float()
                
                # Make sure batch size matches
                if state.size(0) != batch_size:
                    state = state.repeat(batch_size, 1)
                
                # Average state to a single value if needed
                if state.size(1) != 1:
                    state = state.mean(dim=1, keepdim=True)
                
                # Process combined input
                combined = torch.cat([x, state], dim=1)
                return self.q_network(combined)
            else:
                # Just process x
                return self.q_network(x)




