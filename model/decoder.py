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

        self.rnn = nn.LSTM(
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

    def forward(self, img_features, targets, start_input, teacher_forcing_ratio=0.5):
        '''
        :param img_features: tensor of [num_pixels, B, C]
        :param targets: tensor of [T, B, V], each target <end> at the end of the word
        :param start_input: tensor of [1, B, V]
        :return:
            outputs: tensor of [T, B, V]
            weights: tensor of [T, B, num_pixels]
        '''

        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)
        max_length = targets.size(0)

        targets = targets.float()
        rnn_input = start_input.float() # [1, B, V]
        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device) 

        for t in range(max_length):
            context, weight = self.attention(hidden, img_features) # [1, B, C], [num_pixels, B, 1]
            output, (hidden, cell_state) = self.rnn(torch.cat((rnn_input, context), -1), (hidden, cell_state))
            output = self.character_distribution(output)

            outputs[[t]] = output
            weights[[t]] = weight.transpose(0, 2)

            teacher_force = random.random() < teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = targets[[t]]
            else:
                rnn_input = output
            
        return outputs, weights
    
    def greedy(self, img_features, start_input, max_length=10):
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)

        rnn_input = start_input.float()

        hidden = self.init_hidden(batch_size).to(img_features.device)
        cell_state = self.init_hidden(batch_size).to(img_features.device)
        
        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device) 

        # pdb.set_trace()
        for t in range(max_length):
            context, weight = self.attention(hidden, img_features) # [1, B, C], [num_pixels, B, 1]

            rnn_input = torch.cat((rnn_input, context), -1)

            output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))
            output = self.character_distribution(output)

            outputs[[t]] = output
            weights[[t]] = weight.transpose(0, 2)

            rnn_input = output

        return outputs, weights
    
    def beamsearch(self, img_features, start_input, max_length=10, beam_size=3):
        num_pixels = img_features.size(0)
        batch_size = img_features.size(1)

        rnn_input = start_input

        hidden = self.init_hidden(batch_size).to(img_features.device)

        outputs = torch.zeros(max_length, batch_size, self.vocab_size, device=img_features.device)
        weights = torch.zeros(max_length, batch_size, num_pixels, device=img_features.device)
        
        terminal_words, prev_top_words, next_top_words = [], [], []
        prev_top_words.append(rnn_input)

        pdb.set_trace()
        for t in range(max_length):
            for word in prev_top_words:
                context, weight = self.attention(hidden, img_features) # [1, B, C], [num_pixels, B, 1]

                rnn_input = torch.cat((rnn_input, context), -1)

                output, hidden = self.rnn(rnn_input, hidden)
                output = self.character_distribution(output)
                
                topv, topi = output.topk(beam_size, -1)
#                 if topi==10:
                    
#                 term, top = word.addTopk(topi, topv, decoder_hidden, beam_size, voc)
#                 terminal_words.extend(term)
#                 next_top_words.extend(top)

            next_top_words.sort(key=lambda s: s[1], reverse=True)
            prev_top_words = next_top_words[:beam_size]
            next_top_words = []
        terminal_words += [word.toWordScore(voc) for word in prev_top_words]
        terminal_words.sort(key=lambda x: x[1], reverse=True)

        n = min(len(terminal_words), 15)
        return terminal_words[:n]
