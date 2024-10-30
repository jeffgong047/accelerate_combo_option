import torch
import torch.nn as nn
import torch.nn.functional as F

# class BiAttentionClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, bidirectional=True):
#         super(BiAttentionClassifier, self).__init__()
#
#         # Bi-directional LSTM (or GRU) to process the input features
#         # self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)
#         self.rnn = nn.Linear(input_size,hidden_size)
#         bidirectional = False
#         # Attention layer to compute attention for each input vector
#         if bidirectional:
#             self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)  # Applies attention to all input vectors
#         else:
#             self.attention = nn.Linear(hidden_size, hidden_size)
#
#         # Layer normalization
#         if bidirectional:
#             self.layer_norm = nn.LayerNorm(hidden_size * 2)
#         else:
#             self.layer_norm = nn.LayerNorm(hidden_size)
#
#         self.fc = nn.Linear(hidden_size,2)
#         # self.fc = nn.Linear(hidden_size,1)
#         # # Final classifier layer for each input vector
#         # if bidirectional:
#         #     self.fc = nn.Linear(hidden_size * 2, 2)
#         # else:
#         #     self.fc = nn.Linear(hidden_size, 1)
#
#     def check_attention_score(self,x):
#         rnn_output = self.rnn(x)
#         attn_weights = self.attention_net(rnn_output,check_attention_sore=True)
#         return attn_weights
#
#
#     def attention_net(self, rnn_output,check_attention_sore = False):
#         #how to ensure that this function could only be called if check_attention_score is manipulated by check_attention_score function??
#         # rnn_output: [batch_size, seq_len, hidden_size * 2] for Bi-LSTM
#
#         # Compute attention scores for each input vector
#         attn_weights = self.attention(rnn_output)  # [batch_size, seq_len, hidden_size * 2]
#
#         # Softmax across the sequence dimension to get attention weights for each input vector
#         attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, hidden_size * 2]
#
#         # Multiply attention weights with RNN output (element-wise multiplication for each input vector)
#         attended_output = attn_weights * rnn_output  # [batch_size, seq_len, hidden_size * 2]
#
#         # attended_output = torch.bmm(attn_weights.transpose(1, 2), rnn_output)  # [batch_size, 1, hidden_size]
#         #
#         # # Remove the singleton dimension (because we want output size [batch_size, hidden_size])
#         # attended_output = attended_output.squeeze(1)
#
#         if check_attention_sore:
#             return attn_weights
#         else:
#             return attended_output  # This is now the context representation for each input vector
#
#     def forward(self, x):
#         # x: [batch_size, seq_len, input_size]
#
#         # Pass through Bi-LSTM
#         #rnn_output, _ = self.rnn(x)  # [batch_size, seq_len, hidden_size * 2] if bidirectional
#         rnn_output = self.rnn(x)
#         # Apply attention mechanism
#         attended_output = self.attention_net(rnn_output)  # [batch_size, seq_len, hidden_size * 2]
#
#         # Residual connection + layer normalization
#         attended_output = self.layer_norm(attended_output + rnn_output)  # Apply residual connection
#
#         # # Apply the classification layer to each input vector
#         output = self.fc(attended_output)  # [batch_size, seq_len, num_classes]
#
#         return output  # Output: classification result for each input vector


import torch
import torch.nn as nn
import torch.nn.functional as F

class BiAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bidirectional=True):
        super(BiAttentionClassifier, self).__init__()

        # Linear layer to replace RNN (simulating hidden state computation)
        self.linear = nn.Linear(input_size, hidden_size)
        bidirectional = False  # Simplify to single-direction attention

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Final classifier layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def attention_net(self, rnn_output, check_attention_score=False):
        # rnn_output: [batch_size, seq_len, hidden_size]

        # Compute attention scores using dot product (Q · K)
        # Here, we use rnn_output as both Query (Q) and Key (K)
        attn_scores = torch.bmm(rnn_output, rnn_output.transpose(1, 2))  # [batch_size, seq_len, seq_len]

        # Normalize attention scores across the sequence dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Multiply attention weights with the RNN output (to produce context vectors)
        attended_output = torch.bmm(attn_weights, rnn_output)  # [batch_size, seq_len, hidden_size]

        if check_attention_score:
            return attn_weights  # Return attention weights for visualization
        else:
            return attended_output  # Return the attended output (context vectors)

    def check_attention_score(self, x):
        rnn_output = self.linear(x)  # Compute hidden states
        attn_weights = self.attention_net(rnn_output, check_attention_score=True)
        return attn_weights

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]

        # Compute hidden states with linear layer (simulating RNN step)
        rnn_output = self.linear(x)  # [batch_size, seq_len, hidden_size]

        # Apply attention mechanism
        attended_output = self.attention_net(rnn_output)  # [batch_size, seq_len, hidden_size]

        # Apply layer normalization + residual connection
        attended_output = self.layer_norm(attended_output + rnn_output)  # Residual connection

        # Final classification layer
        output = self.fc(attended_output)  # [batch_size, seq_len, num_classes]

        return output
