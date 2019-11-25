import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pdb
import random

class Encoder(nn.Module):
    '''
    Input: Batch image [batch_size, channels, height, width]
    Output: Features [batch_size, channels, feature_height, feature_width]
    '''    
    def __init__(self, depth, n_blocks, growth_rate):
        super(Encoder, self).__init__()

        self.cnn = torchvision.models.DenseNet(
            growth_rate=growth_rate,
            block_config=[depth]*n_blocks
        ).features

        # TODO: fix me
        self.n_features = self.cnn.norm5.num_features
    
    def forward(self, inputs):
        '''
        :param inputs: [B, C, H, W]
        :returms: [num_pixels, B, C']
        '''
        batch_size = inputs.size(0)
        outputs = self.cnn(inputs) # [B, C', H', W']
        outputs = outputs.view(batch_size, self.n_features, -1) # [B, C', H' x W'] == [B, C', num_pixels]
        outputs = outputs.permute(2, 0, 1) # [num_pixels, B, C']
        return outputs

class Attention(nn.Module):
    '''
    Inputs:
        last_hidden: [batch_size, hidden_size, 1]
        encoder_outputs: [batch_size, max_time, feature_size]
    Returns:
        attention_weights: [batch_size, max_time, 1]
    '''
    def __init__(self, feature_size, hidden_size, attn_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(feature_size, attn_size, bias=False)
        self.Ua = nn.Linear(hidden_size, attn_size, bias=False)
        self.va = nn.Linear(attn_size, 1, bias=False)

    def forward(self, last_hidden, encoder_outputs):
        '''
        Input:
        :param last_hidden: [1, B, H]
        :param encoder_outputs: [T, B, C]
        Output:
        weights: [T, B, 1]
        '''
        attention_energies = self._score(last_hidden, encoder_outputs) # [T, B, 1]

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs):
        '''
        Computes an attention score
        :param last_hidden: [batch_size, hidden_dim, 1]
        :param encoder_outputs: [batch_size, max_time, feature_size]
        :return score: [T, B, 1]
        '''

        out = torch.tanh(self.Wa(encoder_outputs) + self.Ua(last_hidden)) # [T, B, A]
        return self.va(out) # [T, B, 1]

class AttnDecoder(nn.Module):
    '''
    Decode one character at a time
    '''
    def __init__(self, feature_size, hidden_size, vocab_size, attn_size):
        super(AttnDecoder, self).__init__()

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

        #self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None
        self.character_distribution = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, input_one_hot, prev_hidden, encoder_outputs):
        '''
        :param input_one_hot: [1, B, V]
        :param prev_hidden: [1, B, H]
        :param encoder_outputs: [T, B, C]
        :return: output [1, B, V], prev_hidden [1, B, H], weights [T, B, 1]
        '''

        # Attention weights
        weights = self.attention(prev_hidden, encoder_outputs) # [T, B, 1]
        context = (weights * encoder_outputs).sum(0, keepdim=True) # [1, B, C]

        # embed characters
        rnn_input = torch.cat((input_one_hot, context), -1) # [1, B, V+C]

        outputs, hidden = self.rnn(rnn_input, prev_hidden) # [1, B, H], [1, B, H]

        output = self.character_distribution(outputs) # [1, B, V]

        output = F.relu(output)

        output = F.log_softmax(output, -1)

        return output, hidden, weights



class Model(nn.Module):
    def __init__(self, depth, n_blocks, growth_rate, hidden_size, attn_size, device, vocab_size, SOS_int, PAD_int, EOS_int):
        super(Model, self).__init__()

        self.encoder = Encoder(depth, n_blocks, growth_rate)
        
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.device = device
        self.vocab_size = vocab_size
        self.feature_size = self.encoder.n_features

        self.decoder = AttnDecoder(
            self.feature_size,
            self.hidden_size,
            self.vocab_size,
            self.attn_size)

        self.SOS_int = SOS_int
        self.EOS_int = EOS_int
        self.PAD_int = PAD_int

    def forward(self, inputs, max_length=15, targets=None, targets_lengths=None, teacher_forcing_ratio=0.5):
        '''
        Input:
        :param inputs: [B, C, H, W]
        :param targets: [max_T, B, V]
        :param targets_lengths: [B, 1]
        Output:
        :outputs: [T, B, V]
        :weights: [num_pixels, B, 1]
        :lengths: [B, 1]
        '''

        encoder_outputs = self.encoder(inputs) # [num_pixels, B, C']

        num_pixels = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        if self.training:
            assert targets is not None and targets_lengths is not None

        decoded_lengths = targets_lengths.squeeze().tolist()
        decoder_input = torch.zeros(1, batch_size, self.vocab_size, device=encoder_outputs.device, dtype=torch.float)
        decoder_input[:,:,self.SOS_int] = 1

        decoder_hidden = self.decoder.init_hidden(batch_size).to(encoder_outputs.device)

        outputs = torch.zeros(max(decoded_lengths), batch_size, self.vocab_size, device=encoder_outputs.device)
        weights = torch.zeros(max(decoded_lengths), batch_size, num_pixels, device=encoder_outputs.device) 

        # pdb.set_trace()
        for t in range(max_length):
            batch_size_t = sum([l > t for l in decoded_lengths])

            if batch_size_t == 0:
                break

            output, decoder_hidden, weight = self.decoder(
                decoder_input[:,:batch_size_t],
                decoder_hidden[:,:batch_size_t],
                encoder_outputs[:,:batch_size_t]
            )
            # output: [1, batch_size_t, V]
            # hidden: [1, batch_size_t, H]
            # weight: [num_pixels, batch_size_t, 1]

            outputs[[t], :batch_size_t, :] = output
            weights[[t], :batch_size_t, :] = weight.transpose(0, 2)

            teacher_force = random.random() < teacher_forcing_ratio
            if self.training and teacher_force:
                decoder_input = targets[[t], :batch_size_t].float() # [1, B, V]
            else:
                decoder_input = output

        return outputs, weights, torch.tensor(decoded_lengths).unsqueeze(-1)