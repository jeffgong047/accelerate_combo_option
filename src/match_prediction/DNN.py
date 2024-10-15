import torch
import torch.nn as nn
import torch.nn.functional as F

class BiAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bidirectional=True):
        super(BiAttentionClassifier, self).__init__()

        # Bi-directional LSTM (or GRU) to process the input features
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)

        # Attention layer to compute attention for each input vector
        if bidirectional:
            self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)  # Applies attention to all input vectors
        else:
            self.attention = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        if bidirectional:
            self.layer_norm = nn.LayerNorm(hidden_size * 2)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # Final classifier layer for each input vector
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def attention_net(self, rnn_output):
        # rnn_output: [batch_size, seq_len, hidden_size * 2] for Bi-LSTM

        # Compute attention scores for each input vector
        attn_weights = self.attention(rnn_output)  # [batch_size, seq_len, hidden_size * 2]

        # Softmax across the sequence dimension to get attention weights for each input vector
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, hidden_size * 2]

        # Multiply attention weights with RNN output (element-wise multiplication for each input vector)
        attended_output = attn_weights * rnn_output  # [batch_size, seq_len, hidden_size * 2]

        return attended_output  # This is now the context representation for each input vector

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]

        # Pass through Bi-LSTM
        rnn_output, _ = self.rnn(x)  # [batch_size, seq_len, hidden_size * 2] if bidirectional

        # Apply attention mechanism
        attended_output = self.attention_net(rnn_output)  # [batch_size, seq_len, hidden_size * 2]

        # Residual connection + layer normalization
        attended_output = self.layer_norm(attended_output + rnn_output)  # Apply residual connection

        # Apply the classification layer to each input vector
        output = self.fc(attended_output)  # [batch_size, seq_len, num_classes]

        return output  # Output: classification result for each input vector



