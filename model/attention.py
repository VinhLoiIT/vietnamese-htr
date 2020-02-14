import torch
import torch.nn as nn
import torch.nn.functional as F

def get_attention(attn_type, feature_size, hidden_size, attn_size, nhead=1):
    if nhead > 1:
        assert attn_size % nhead == 0
        if attn_type == 'scale_dot_product':
            attn = ScaleDotProductMultiHeadAttention(nhead)
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
        self.Wa = nn.Linear(queries_size, attn_size)
        self.Ua = nn.Linear(key_size, attn_size)
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
        queries = self.Wa(queries).transpose(0,1) # [B,T,A]
        keys = self.Ua(keys).transpose(0,1) # [B,S,A]
        
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
    def __init__(self, attn, nhead=1):
        super().__init__()
        self.attn = attn
        self.nhead = nhead

    def apply_mask(self, weights, attn_mask):
        '''
        Shapes:
        - weights: [B*nhead,T,S]
        - attn_mask: [B,T,S]
        - output: [B,T,S]
        '''
        if attn_mask is not None:
            saved_shape = weights.shape
            weights = weights.view(self.nhead, *attn_mask.shape) # [nhead, B,T,S]
            attn_mask = attn_mask.expand_as(weights) # [nhead, B,T,S]
            weights[~attn_mask] = float('-inf')
            weights = weights.view(*saved_shape) # [B*nhead,T,S]
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

        queries_multihead = queries.view(len_queries, batch_size*self.nhead, -1) # [T, B*nhead, head_attn_size]
        keys_multihead = keys.view(len_keys, batch_size*self.nhead, -1) # [S, B*nhead, head_attn_size]

        weights = self.attn.score(queries_multihead, keys_multihead) # [B*nhead, T, S]
        weights = self.apply_mask(weights, attn_mask)
        weights = F.softmax(weights, dim=-1) # [B,T,S]
        
        values = weights.bmm(keys_multihead.transpose(0,1)) # [B*nhead, T, head_attn_size]
        values = values.view(len_queries, batch_size, -1)
        
        if output_weights:
            weights = weights.view(self.nhead, batch_size, len_queries, -1) # [nhead, B, T, S]
            weights = torch.mean(weights, dim=0, keepdim=False) # weight: [B, T, S]
            return values, weights
        else:
            return values, None

class AdditiveMultiHeadAttention(MultiHeadAttention):
    def __init__(self, attn_size, nhead):
        assert attn_size % nhead == 0
        head_attn_size = attn_size // nhead
        attn = AdditiveAttention(head_attn_size, head_attn_size, head_attn_size)
        super().__init__(attn, nhead)

class ScaleDotProductMultiHeadAttention(MultiHeadAttention):
    def __init__(self, nhead):
        attn = ScaleDotProductAttention()
        super().__init__(attn, nhead)
