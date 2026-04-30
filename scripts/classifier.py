import torch
import torch.nn as nn
import copy
import math
import itertools

def clones(module, N):
    """
    From python's copy module: "A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original."
    nn.ModuleList holds submodules in a list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_alibi(attention_heads, num_patches_H, num_patches_W):
    """
    Function adapted to rectangular-shaped input from original CROMA paper by Fuller et al. (2023)
    """
    points = list(itertools.product(range(num_patches_H), range(num_patches_W)))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(attention_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, attention_heads, num_patches_H, num_patches_W)

def attention(query, key, value, alibi, mask=None, dropout=None):
    """
    query.size() == (batch, heads, seq_len, d_k)
    scores.size() == (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if alibi:
        scores = scores + alibi
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Adapted from the annotated transformers. Took out query,key,val parameters in forward method because we don't need cross-attention. Used alibi positional embeddings. They're defined here in a clunky way, which requires num_patches_H, num_patches_W (size of spectrogram). I should change them. 
    """
    def __init__(self, d_model, number_heads, num_patches_H, num_patches_W, alibi_bias, dropout=0.1):
        super().__init__()
        assert d_model % number_heads == 0
        self.h = number_heads
        self.d_k = d_model // number_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        if alibi_bias:
            self.alibi = get_alibi(number_heads, num_patches_H=num_patches_H, num_patches_W=num_patches_W)
        else:
            self.alibi = None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """
        Multi-Head attention adds an extra dimension, h, to our residual stream, which had dimensions [n_batches, seq_len, d_model]
        """
        if mask is not None:
            mask = mask.unsqueeze(1) # Add number_heads to the mask
        
        n_batches = x.size(0)

        query, key, value = [lin(x).view(n_batches, -1, self.h, self.d_k)\
                             .transpose(1, 2) for lin in self.linears[:3]]
                             # projections from d_model to d_model, then splits and transposes
                             # query.size() == [n_batches, h, seq_len, d_k]

        # returns tensor of size (batch, heads, seq_len, d_k)
        x, self.attn = attention(query, key, value, self.alibi, mask=mask, dropout=self.dropout)

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
    def __init__(self, d_model, number_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model=d_model, number_heads=number_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, alibi_bias=None, mask=None):
        x = x + self.drop1(self.attn(self.ln1(x), ))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x