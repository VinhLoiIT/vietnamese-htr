import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, vocab_size, attn_size):
        super(Decoder, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_size = attn_size

        self.rnn = nn.GRU(
            input_size=self.vocab_size+self.feature_size,
            hidden_size=self.hidden_size,
        )

        self.attention = Attention(
            self.feature_size,
            self.hidden_size,
            self.attn_size)

        self.character_distribution = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, img_features, targets, teacher_forcing_ratio=0.5):
        '''
        :param img_features: tensor of [num_pixels, B, C]
        :param targets: tensor of [T, B, V], each target has <start> and <end> at begin and end of the word
        :return:
            outputs: tensor of [T, B, V]
            weights: tensor of [T, B, num_pixels]
        '''

        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)
        max_length = targets.size(0)

        targets = targets.float()
        rnn_input = targets[[0]].float() # [1, B, V]
        hidden = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device) 

        # pdb.set_trace()
        for t in range(max_length - 1):
            context, weight = self.attention(hidden, img_features) # [1, B, C], [num_pixels, B, 1]

            teacher_force = random.random() < teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = torch.cat((targets[[t]], context), -1)
            else:
                rnn_input = torch.cat((rnn_input, context), -1)

            output, hidden = self.rnn(rnn_input, hidden)
            output = self.character_distribution(output)

            outputs[[t]] = output
            weights[[t]] = weight.transpose(0, 2)
            
            rnn_input = output
            
        return outputs, weights