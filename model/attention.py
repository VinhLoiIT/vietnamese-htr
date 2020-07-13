import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def get_attention(attn_type, attn_size, nhead=None):
    if nhead is None:
        if attn_type == 'additive':
            return AdditiveAttention(attn_size)
        if attn_type == 'scaledotproduct':
            return ScaleDotProductAttention(attn_size)
    else:
        assert attn_size % nhead == 0, 'attn_size = {} should devisible to number of heads = {}'.format(attn_size, nhead)
        head_size = attn_size // nhead
        if attn_type == 'additive':
            attn = AdditiveAttention(head_size)
            attn = MultiHeadAttention(attn, nhead)
            return attn
        elif attn_type == 'scaledotproduct':
            attn = ScaleDotProductAttention(head_size)
            attn = MultiHeadAttention(attn, nhead)
            return attn
    raise ValueError('Invalid type of attention, should be {}'.format(['additive', 'scaledotproduct']))

class Attention(nn.Module):
    def __init__(self, attn_size):
        super(Attention, self).__init__()
        self.attn_size = attn_size

    def score(self, queries, keys):
        raise NotImplementedError()

    def forward(self, queries, keys, values, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [B, T, A]
        :param keys: [B, S, A]
        :param values: [B, S, A]
        :param attn_mask: [B,T,S]
        Output:
        - values: [B, T, C]
        - weights: [B, T, S] if output_weights = True else None
        '''
        weights = self.score(queries, keys) # [B,T,S]
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            weights += attn_mask
        weights = F.softmax(weights, dim=-1)
        values = weights.bmm(values) # [B, T, A]
        if output_weights:
            return values, weights
        else:
            return values, None

class AdditiveAttention(Attention):
    def __init__(self, attn_size):
        super().__init__(attn_size)
        self.Wa = nn.Linear(attn_size, attn_size)
        self.Ua = nn.Linear(attn_size, attn_size)
        self.va = nn.Linear(attn_size, 1)

    def score(self, queries, keys):
        '''
        Input:
        - queries: [B, T, A]
        - keys: [B, S, A]
        - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
        Output:
        - weights: [B, T, S]
        '''
        keys = self.Wa(keys) # [B,S,A]
        queries = self.Ua(queries) # [B,T,A]

        keys = keys.unsqueeze(1) # [B,1,S,A]
        queries = queries.unsqueeze(2) # [B,T,1,A]

        weights = self.va(torch.tanh(queries + keys)) # [B,T,S,1]
        weights = weights.squeeze(-1) # [B,T,S]
        return weights

class ScaleDotProductAttention(Attention):
    def __init__(self, attn_size):
        super().__init__(attn_size)

    def score(self, queries, keys):
        '''
        Input:
        - queries: [B, T, A]
        - keys: [B, S, A]
        Output:
        - weights: [B, T, S]
        '''
        attn_dim = queries.size(-1)
        # [B,T,A] x [B,A,S] = [B,T,S]
        matmul = queries.bmm(keys.transpose(1, 2))
        scaled = matmul / attn_dim # [B,T,S]

        return scaled

class MultiHeadAttention(nn.Module):
    def __init__(self, attn, nhead):
        super().__init__()
        self.head_size = attn.attn_size
        self.nhead = nhead
        self.attn_size = self.nhead * self.head_size

        self.heads = _get_clones(attn, nhead)
        self.q_proj = _get_clones(nn.Linear(self.attn_size, self.head_size), nhead)
        self.k_proj = _get_clones(nn.Linear(self.attn_size, self.head_size), nhead)
        self.v_proj = _get_clones(nn.Linear(self.attn_size, self.head_size), nhead)
        self.o_proj = nn.Linear(self.attn_size, self.attn_size)

    def forward(self, queries, keys, values, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [B, T, A]
        :param keys: [B, S, A]
        Output:
        - values: [B, T, A]
        - weights: [nhead, B, T, S]
        '''
        q_projected = [q_proj(queries) for q_proj in self.q_proj]
        k_projected = [k_proj(keys) for k_proj in self.k_proj]
        v_projected = [v_proj(values) for v_proj in self.v_proj]

        head_outputs = [head(q,k,v,attn_mask, output_weights) for head,q,k,v in zip(self.heads, q_projected, k_projected, v_projected)]
        values, weights = list(zip(*head_outputs))
        # values (list): nhead * [B,T,head_attn_size]
        # weights (list): nhead * [B,T,S]

        values = torch.cat(values, -1) # [B,T,A]
        values = self.o_proj(values) # [B,T,A]
        if output_weights:
            weights = torch.stack(weights, dim=0) # [nhead,B,T,S]
            # weights = torch.mean(weights, dim=0, keepdim=False) # weight: [B, T, S]
            return values, weights
        else:
            return values, None

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
