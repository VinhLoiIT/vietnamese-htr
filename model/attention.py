import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def get_attention(attn_type, feature_size, hidden_size, attn_size, nhead=1):
    if nhead > 1:
        assert attn_size % nhead == 0
        if attn_type == 'scale_dot_product':
            attn = ScaleDotProductMultiHeadAttention(attn_size, nhead)
        elif attn_type == 'additive':
            attn = AdditiveMultiHeadAttention(attn_size, nhead)
        else:
            raise ValueError('Unknow attn_type={}, should be {}'.format(attn_type, ['scale_dot_product', 'additive']))
    else:
        if attn_type == 'scale_dot_product':
            attn = ScaleDotProductAttention()
        elif attn_type == 'additive':
            attn = AdditiveAttention(hidden_size, feature_size, attn_size)
        else:
            raise ValueError('Unknow attn_type={}, should be {}'.format(attn_type, ['scale_dot_product', 'additive']))
    
    return attn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
    def score(self, queries, keys):
        raise NotImplementedError()
    
    def apply_mask(self, weights, attn_mask):
        if attn_mask is not None:
            weights[~attn_mask] = float('-inf')

        return weights

    def forward(self, queries, keys, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [T, B, H]
        :param keys: [S, B, C]
        Output:
        - values: [T, B, C]
        - weights: [B, T, S] if output_weights = True else None
        '''
        weights = self.score(queries, keys) # [B,T,S]
        weights = self.apply_mask(weights, attn_mask) # [B,T,S]
        weights = F.softmax(weights, dim=-1)
        values = weights.bmm(keys.transpose(0,1)) # [B, T, C]
        values = values.transpose(0,1)
        if output_weights:
            return values, weights
        else:
            return values, None

class AdditiveAttention(Attention):
    def __init__(self, queries_size, key_size, attn_size):
        super().__init__()
        self.Wa = nn.Linear(key_size, attn_size)
        self.Ua = nn.Linear(queries_size, attn_size)
        self.va = nn.Linear(attn_size, 1)

    def score(self, queries, keys):
        '''
        Input:
        - queries: [T, B, A]
        - keys: [S, B, A]
        - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
        Output:
        - weights: [B, T, S]
        '''
        keys = self.Wa(keys).transpose(0,1) # [B,S,A]
        queries = self.Ua(queries).transpose(0,1) # [B,T,A]
        
        keys = keys.unsqueeze(1) # [B,1,S,A]
        queries = queries.unsqueeze(2) # [B,T,1,A]

        weights = self.va(torch.tanh(queries + keys)) # [B,T,S,1]
        weights = weights.squeeze(-1) # [B,T,S]
        return weights

class ScaleDotProductAttention(Attention):
    def __init__(self):
        super().__init__()

    def score(self, queries, keys):
        '''
        Input:
        - queries: [T, B, A] - Keys, Values
        - keys: [S, B, A] - Queryes
        Output:
        - weights: [B, T, S]
        '''
        attn_dim = queries.size(-1)
        queries = queries.transpose(0, 1) # [B,T,A]
        keys = keys.transpose(0, 1) # [B,S,A]

        # [B,T,A] x [B,A,S] = [B,T,S]
        matmul = queries.bmm(keys.transpose(1, 2))
        scaled = matmul / attn_dim # [B,T,S]
        
        return scaled


class MultiHeadAttention(Attention):
    def __init__(self, attn, attn_size, nhead=1):
        super().__init__()
        assert attn_size % nhead == 0
        head_attn_size = attn_size // nhead
        self.nhead = nhead
        self.heads = _get_clones(attn, self.nhead)
        self.q_proj = nn.Linear(head_attn_size, head_attn_size)
        self.k_proj = nn.Linear(head_attn_size, head_attn_size)
        self.v_proj = nn.Linear(head_attn_size, head_attn_size)
        self.o_proj = nn.Linear(attn_size, attn_size)

    def apply_mask(self, weights, attn_mask):
        '''
        Shapes:
        - weights: [nhead,B,T,S]
        - attn_mask: [B,T,S]
        - output: [nhead,B,T,S]
        '''
        if attn_mask is not None:
            attn_mask = attn_mask.expand_as(weights) # [nhead,B,T,S]
            super().apply_mask(weights, attn_mask) # [nhead, B,T,S]
        return weights

    def forward(self, queries, keys, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [T, B, A]
        :param keys: [S, B, A]
        Output:
        - values: [T, B, A]
        - weights: [B, T, S]
        '''
        batch_size = queries.size(1)
        len_queries = queries.size(0)
        len_keys = keys.size(0)
        attn_size = keys.size(-1)
        assert attn_size % self.nhead == 0

        queries_multihead = queries.view(len_queries, batch_size, self.nhead, -1).permute(2,0,1,3) # [nhead, T, B, head_attn_size]
        keys_multihead = keys.view(len_keys, batch_size, self.nhead, -1).permute(2,0,1,3) # [nhead, S, B, head_attn_size]
        
        queries_multihead_proj = self.q_proj(queries_multihead) # [nhead, T, B, head_attn_size]
        keys_multihead_proj = self.k_proj(keys_multihead) # [nhead, S, B, head_attn_size]
        values_multihead_proj = self.v_proj(keys_multihead) # [nhead, S, B, head_attn_size] - we usually pass values=keys

        weights = torch.cat([self.heads[i].score(queries_multihead[i], keys_multihead[i]).unsqueeze(0) for i in range(self.nhead)], dim=0) # [nhead, B, T, S]
        weights = self.apply_mask(weights, attn_mask) # [nhead, B,T,S]
        weights = F.softmax(weights, dim=-1) # [nhead, B,T,S]
        
        # TODO: add dropout to weights..
        
        values = weights.matmul(values_multihead_proj.transpose(1,2)) # [nhead, B,T,head_attn_size]
        values = values.transpose(0,2).reshape(len_queries, batch_size, -1) # [T,B,A]
        values = self.o_proj(values)
        
        if output_weights:
            weights = torch.mean(weights, dim=0, keepdim=False) # weight: [B, T, S]
            return values, weights
        else:
            return values, None

class AdditiveMultiHeadAttention(MultiHeadAttention):
    def __init__(self, attn_size, nhead):
        assert attn_size % nhead == 0
        head_attn_size = attn_size // nhead
        attn = AdditiveAttention(head_attn_size, head_attn_size, head_attn_size)
        super().__init__(attn, attn_size, nhead)

class ScaleDotProductMultiHeadAttention(MultiHeadAttention):
    def __init__(self, attn_size, nhead):
        assert attn_size % nhead == 0
        attn = ScaleDotProductAttention()
        super().__init__(attn, attn_size, nhead)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
