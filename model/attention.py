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
