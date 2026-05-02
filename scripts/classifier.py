import torch
import torch.nn as nn
import math
from utils import clones

def attention(query, key, value, alibi, padding_mask=None, dropout=None):
    """
    query.size() == (batch, heads, seq_len, d_k)
    scores.size() == (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if alibi is not None:
        scores = scores + alibi
    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Adapted from the annotated transformers. Took out query,key,val parameters in forward method because we don't need cross-attention. Used alibi positional embeddings. They're defined here in a clunky way, which requires num_patches_H, num_patches_W (size of spectrogram). I should change them. 
    """
    def __init__(self, d_model, number_heads, alibi_bias=None, dropout=0.1):
        super().__init__()
        assert d_model % number_heads == 0
        self.h = number_heads
        self.d_k = d_model // number_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.alibi_bias = alibi_bias
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask=None):
        """
        Multi-Head attention adds an extra dimension, h, to our residual stream, which had dimensions [n_batches, seq_len, d_model]
        """
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1) # Add number_heads to the mask
        
        n_batches = x.size(0)

        query, key, value = [lin(x).view(n_batches, -1, self.h, self.d_k)\
                             .transpose(1, 2) for lin in self.linears[:3]]
                             # projections from d_model to d_model, then splits and transposes
                             # query.size() == [n_batches, h, seq_len, d_k]

        # returns tensor of size (batch, heads, seq_len, d_k)
        x, self.attn = attention(query, key, value, self.alibi_bias, padding_mask=padding_mask, dropout=self.dropout)

        x = (
            x.transpose(1, 2)
            .contiguous() # Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor.
            .view(n_batches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
        
class TransformerStack(nn.Module):
    """
    Manage alibi parameters such as grid size
    """
    def __init__(self, d_model, number_heads, bias, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model=d_model, number_heads=number_heads, alibi_bias=bias, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.drop1(self.attn(self.ln1(x), padding_mask))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x