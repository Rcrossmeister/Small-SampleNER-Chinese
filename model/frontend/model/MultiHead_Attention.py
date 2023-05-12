import torch
import torch.nn as nn


class MultiHead_SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHead_SelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None): # x.shape: [batch_size, seq_length, features]
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Linearly project the queries, keys, and values
        query_proj = self.query_linear(x) # query.shape: [batch_size, seq_length, features]
        key_proj = self.key_linear(x)
        value_proj = self.value_linear(x)

        # Reshape the projected queries, keys, and values
        query_proj = query_proj.view(batch_size, -1, self.num_heads, self.head_dim)
        # query.shape: [batch_size, seq_length, num_head, head_dim]
        key_proj = key_proj.view(batch_size, -1, self.num_heads, self.head_dim)
        value_proj = value_proj.view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose the dimensions of the reshaped queries, keys, and values
        query_proj = query_proj.transpose(1, 2) # query.shape: [batch_size, num_head, seq_length, head_dim]
        key_proj = key_proj.transpose(1, 2)
        value_proj = value_proj.transpose(1, 2)

        # Compute the attention scores
        scores = torch.matmul(query_proj, key_proj.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        # scores.shape: [batch_size, num_head, seq_length, seq_length]
        # Apply the mask, if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply the softmax function to the attention scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to the attention weights
        attention_weights = nn.Dropout(0.1)(attention_weights)

        # Compute the context vector
        context = torch.matmul(attention_weights, value_proj)
        # context.shape: [batch_size, num_head, seq_length, head_dim]

        # Reshape and transpose the context vector
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Linearly project the context vector
        output = self.output_linear(context)

        return output
