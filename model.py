import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
    
    def forward(self, x):
        return self.cnn(x)

class Attention(nn.Module):
    '''
    Inputs:
        last_hidden: [batch_size, hidden_size, 1]
        encoder_outputs: [batch_size, max_time, feature_size]
    Returns:
        attention_weights: [batch_size, max_time, 1]
    '''
    def __init__(self, batch_size, feature_size, hidden_size, attn_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(feature_size, attn_size, bias=False)
        self.Ua = nn.Linear(hidden_size, attn_size, bias=False)
        self.va = nn.Linear(attn_size, 1)

    def forward(self, last_hidden, encoder_outputs):#, seq_len=None):
        '''
        Input:
        :param last_hidden: [1, B, H]
        :param encoder_outputs: [T, B, C]
        Output:
        weights: [T, B, 1]
        '''
        attention_energies = self._score(last_hidden, encoder_outputs) # [T, B, 1]

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs):
        '''
        Computes an attention score
        :param last_hidden: [batch_size, hidden_dim, 1]
        :param encoder_outputs: [batch_size, max_time, feature_size]
        :return score: [T, B, 1]
        '''

        out = F.tanh(self.Wa(encoder_outputs) + self.Ua(last_hidden)) # [T, B, A]
        return self.va(out) # [T, B, 1]

class AttnDecoder(nn.Module):
    '''
    Decode one character at a time
    '''
    def __init__(self, batch_size, feature_size, hidden_size, vocab_size, attn_size):
        super(AttnDecoder, self).__init__()

        self.batch_size = batch_size
        self.encoder_features = feature_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attn_size = attn_size

        self.rnn = nn.GRU(
            input_size=self.vocab_size+self.hidden_size,
            hidden_size=self.hidden_size,
        )

        self.attention = Attention(
            self.batch_size,
            self.feature_size,
            self.hidden_size,
            self.attn_size)

        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None
        self.character_distribution = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)

    def forward(self, input_one_hot, prev_hidden, encoder_outputs):
        '''
        :param input: [1, B, V]
        :param prev_hidden: [1, B, H]
        :param encoder_outputs: [T, B, C]
        :return: output [1, B, V], prev_hidden [1, B, H], weights [T, B, 1]
        '''

        assert input.size() == torch.Size([self.batch_size, 1, self.vocab_size])
        assert prev_hidden.size() == torch.Size([self.batch_size, self.hidden_size, 1])

        # Attention weights
        weights = self.attention(prev_hidden, encoder_outputs) # [T, B, 1]
        context = (weights * encoder_outputs).sum(0, keepdim=True) # [1, B, C]

        # embed characters
        rnn_input = torch.cat((input, context), -1) # [1, B, V+C]

        outputs, hidden = self.rnn(rnn_input, prev_hidden) # [1, B, H], [1, B, H]

        output = self.character_distribution(outputs) # [1, B, V]

        # currently is CrossEntropyLoss, then dont need these line
        # if self.decoder_output_fn:
        #     output = self.decoder_output_fn(output, -1)

        return output, hidden, weights



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.attn_size = config['attn_size']
        self.device = config['device']

        self.encoder = Encoder(config['depth'], config['n_blocks'], config['growth_rate'])
        self.feature_size = self.encoder.n_features

        self.decoder = AttnDecoder(
            self.batch_size,
            self.feature_size,
            self.hidden_size,
            self.vocab_size,
            self.attn_size)


    def forward(self, inputs, targets, targets_lengths):
        '''
        Input:
        :param inputs: [B, C, H, W]
        :param targets: [T, B, V]
        :param targets_lengths: [B, 1]
        Output:
        :outputs: [T, B, V]
        :weights: [T, B, 1]
        '''
        pdb.set_trace()

        encoder_outputs = self.encoder(inputs) # [B, C', H', W']
        encoder_outputs = encoder_outputs.view(self.batch_size, self.feature_size, -1) # [B, C', H' x W'] == [B, C', T]
        encoder_outputs.permute_(2, 0, 1) # [T, B, C']

        max_length = targets.size(0)
        outputs = torch.zeros(max_length, self.batch_size, self.vocab_size).to(self.device)
        weights = torch.zeros(max_length, self.batch_size, 1).to(self.device)

        decoder_input = '<sos>'
        hidden = self.decoder.init_hidden().to(self.device)

        for t in range(max_length):
            output, hidden, weight = self.decoder(
                decoder_input,
                prev_hidden=hidden,
                encoder_outputs=encoder_outputs)

            outputs.append(output)
            weights.append(weight)

            # if teacher_forcing:
            decoder_input = targets[[t]]
            # else:
            #     # TODO
            #     raise NotImplementedError

        return outputs, weights