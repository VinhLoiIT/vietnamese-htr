import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from .attention import Attention, AdditiveAttention, ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, attn, nhead):
        super().__init__()
        self.attn_size = attn.attn_size
        self.nhead = nhead
        assert self.attn_size % self.nhead == 0, 'Attention size = {} must divisible by number of heads = {}'.format(self.attn_size, self.nhead)
        self.head_size = self.attn_size // self.nhead

        self.heads = _get_clones(attn, nhead)
        self.q_proj = _get_clones(nn.Linear(attn_size, self.head_size), nhead))
        self.k_proj = _get_clones(nn.Linear(attn_size, self.head_size), nhead))
        self.v_proj = _get_clones(nn.Linear(attn_size, self.head_size), nhead))
        self.o_proj = nn.Linear(attn_size, attn_size)

    def forward(self, queries, keys, values, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [T, B, A]
        :param keys: [S, B, A]
        Output:
        - values: [T, B, A]
        - weights: [B, T, S]
        '''
        q_projected = [q_proj(queries) for q_proj in self.q_proj]
        k_projected = [k_proj(keys) for k_proj in self.k_proj]
        v_projected = [v_proj(values) for v_proj in self.v_proj]

        head_outputs = []
        for head in self.heads:
            q,k,v = zip(q_projected, k_projected, v_projected)
            head_outputs.append(head(q,k,v,attn_mask, output_weights))

        values, weights = zip(*head_outputs)
        # values (list): nhead * [T,B,head_attn_size]
        # weights (list): nhead * [B,T,S]

        values = torch.cat(values, -1) # [T,B,A]
        values = self.o_proj(values) # [T,B,A]
        if output_weights:
            weights = torch.stack(weights, dim=0) # [nhead,B,T,S]
            weights = torch.mean(weights, dim=0, keepdim=False) # weight: [B, T, S]
            return values, weights
        else:
            return values, None

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
