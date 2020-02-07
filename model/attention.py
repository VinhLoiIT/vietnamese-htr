import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_size, hidden_size, attn_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(feature_size, attn_size)
        self.Ua = nn.Linear(hidden_size, attn_size)
        self.va = nn.Linear(attn_size, 1)

    def forward(self, last_hidden, encoder_outputs):
        '''
        Input:
        :param last_hidden: [1, B, H]
        :param encoder_outputs: [num_pixels, B, C]
        Output:
        context: [1, B, C]
        weights: [num_pixels, B, 1]
        '''
        attn1 = self.Wa(encoder_outputs) # [num_pixels, B, A]
        attn2 = self.Ua(last_hidden) # [1, B, A]
        attn = self.va(torch.tanh(attn1 + attn2)) # [num_pixels, B, 1]

        weights = F.softmax(attn.squeeze(-1), 0).unsqueeze(-1) # [num_pixels, B, 1]
        context = (weights * encoder_outputs).sum(0, keepdim=True) # [1, B, C]

        return context, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, attn_size, nhead=1):
        super().__init__()
        if attn_size % nhead != 0:
            raise ValueError(f'attn_size should divisible nhead: {attn_size} % {nhead} != 0')

        self.Wa = nn.Linear(feature_size, attn_size)
        self.Ua = nn.Linear(hidden_size, attn_size)
        self.nhead = nhead
        self.head_attn_size = attn_size // nhead
        self.attn = Attention(self.head_attn_size, self.head_attn_size, self.head_attn_size)

    def forward(self, last_hidden, encoder_outputs):
        '''
        Input:
        :param last_hidden: [1, B, H]
        :param encoder_outputs: [num_pixels, B, C]
        Output:
        context: [1, B, C]
        weights: [num_pixels, B, 1]
        '''
        encoder_outputs = self.Wa(encoder_outputs) # [num_pixels, B, A]
        last_hidden = self.Ua(last_hidden) # [1, B, A]

        encoder_outputs = encoder_outputs.view(*encoder_outputs.shape[:-1], self.nhead, self.head_attn_size)
        last_hidden = last_hidden.view(*last_hidden.shape[:-1], self.nhead, self.head_attn_size)

        # contexts = []
        # weights = []
        # for t in range(len(last_hidden)):
        #     context, weight = self.attn(last_hidden[[t]], encoder_outputs)
        #     # weight: [num_pixels, B, nhead, 1]
        #     # context: [1, B, nhead, A]
        #     contexts.append(context)
        #     weights.append(weight.transpose(0, 2))
        # context = torch.cat(contexts, dim=0)
        # weight = torch.cat(weights, dim=0)
        
        context, weight = self.attn(last_hidden, encoder_outputs)
        # weight: [num_pixels, B, nhead, 1]
        # context: [1, B, nhead, A]

        weight = torch.mean(weight, dim=-2, keepdim=False) # weight: [num_pixels, B, 1]
        context = context.view(context.shape[0], context.shape[1], -1) # context: [1, B, nhead x head_attn_size] = [1, B, A]
        return context, weight
