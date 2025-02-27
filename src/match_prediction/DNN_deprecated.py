import torch
import torch.nn as nn
import torch.nn.functional as F



class simply_attention_block(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(simply_attention_block, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # self.fc = nn.Linear(hidden_size, num_classes)
    
    def attention_net(self, linear_output, check_attention_score=False):
        # rnn_output: [batch_size, seq_len, hidden_size]

        # Compute attention scores using dot product (Q · K)
        # Here, we use linear_output as both Query (Q) and Key (K)]
        # linear_output has shape [batch_size, seq_len, hidden_size]

        attn_scores = torch.bmm(linear_output, linear_output.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        # Normalize attention scores across the sequence dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        # breakpoint()
        # print(torch.sum(attn_weights.squeeze(0), dim=0))
        # print(torch.sum(attn_weights.squeeze(0),dim=1))
        # print('attention score difference, the squence used attention - attention payed to the sequence ',torch.sum(attn_weights.squeeze(0), dim=0)- torch.sum(attn_weights.squeeze(0),dim=1))
        # Multiply attention weights with the RNN output (to produce context vectors)
        attended_output = torch.bmm(attn_weights, linear_output)  # [batch_size, seq_len, hidden_size]

        if check_attention_score:
            return attn_weights  # Return attention weights for visualization
        else:
            return attended_output  # Return the attended output (context vectors)
    def forward(self, x):
        linear_output = self.linear(x)
        attended_output = self.attention_net(linear_output)
        return self.layer_norm(attended_output + linear_output)


class BiAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1,bidirectional=True):
        super(BiAttentionClassifier, self).__init__()


        # self.linear = nn.Linear(input_size, hidden_size)
        # bidirectional = False  # Simplify to single-direction attention
        # self.dropout = torch.nn.Dropout(0.2)
        # # Layer normalization
        # self.layer_norm = nn.LayerNorm(hidden_size)

        # # Final classifier layer
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.simply_attention_blocks = nn.ModuleList([simply_attention_block(input_size, hidden_size, num_classes)] + [simply_attention_block(hidden_size, hidden_size, num_classes) for _ in range(num_layers-1)])
        self.fc = nn.Linear(hidden_size, num_classes)
                                                     

        self.policy_head = None  # Initialize as None

    def attention_net(self, linear_output, check_attention_score=False):
        # rnn_output: [batch_size, seq_len, hidden_size]

        # Compute attention scores using dot product (Q · K)
        # Here, we use linear_output as both Query (Q) and Key (K)]
        # linear_output has shape [batch_size, seq_len, hidden_size]

        attn_scores = torch.bmm(linear_output, linear_output.transpose(1, 2))  # [batch_size, seq_len, seq_len]
     
        # print('attention score difference, the squence used attention - attention payed to the sequence ',torch.sum(attn_scores.squeeze(0), dim=0)- torch.sum(attn_scores.squeeze(0),dim=1))
        # Normalize attention scores across the sequence dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        # breakpoint()
        # print(torch.sum(attn_weights.squeeze(0), dim=0))
        # print(torch.sum(attn_weights.squeeze(0),dim=1))
        # print('attention score difference, the squence used attention - attention payed to the sequence ',torch.sum(attn_weights.squeeze(0), dim=0)- torch.sum(attn_weights.squeeze(0),dim=1))
        # Multiply attention weights with the RNN output (to produce context vectors)
        attended_output = torch.bmm(attn_weights, linear_output)  # [batch_size, seq_len, hidden_size]

        if check_attention_score:
            return attn_weights  # Return attention weights for visualization
        else:
            return attended_output  # Return the attended output (context vectors)

    def check_attention_score(self, x):
        rnn_output = self.linear(x)  # Compute hidden states
        attn_weights = self.attention_net(rnn_output, check_attention_score=True)
        return attn_weights


    def get_embedding(self, x):
        linear_output = self.linear(x)
        return linear_output
    
    def init_policy_head(self, reinit=False):
        """Initialize or reinitialize the policy head"""
        if self.policy_head is None or reinit:
            self.policy_head = nn.Linear(self.fc.in_features, self.fc.out_features)
            # Initialize with current classifier weights
            with torch.no_grad():
                self.policy_head.weight.copy_(self.fc.weight)
                self.policy_head.bias.copy_(self.fc.bias)

    def get_policy_prediction(self, x):
        """Get prediction using the policy head"""
        if self.policy_head is None:
            self.init_policy_head()
            
        # Use the shared feature extraction pipeline
        linear_output = self.linear(x)
        attended_output = self.attention_net(linear_output)
        attended_output = self.layer_norm(attended_output + linear_output)
        
        # Use policy head instead of classifier
        return self.policy_head(attended_output)

    def forward(self, x, use_policy_head=False):
        """Forward pass with option to use policy head"""
        if use_policy_head:
            return self.get_policy_prediction(x)
        
        # Original classifier forward pass
        # linear_output = self.linear(x)
        # attended_output = self.attention_net(linear_output)
        # attended_output = self.layer_norm(attended_output + linear_output)
        # return self.fc(attended_output)
    def forward(self, x):
        for block in self.simply_attention_blocks:
            x = block(x)
        return self.fc(x)




