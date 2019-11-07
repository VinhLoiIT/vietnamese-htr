import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Encoder(nn.Module):
    def __init__(self, depth=4, n_blocks=3, growth_rate=96):
        super(Encoder, self).__init__()
        self.cnn = torchvision.models.DenseNet(
            growth_rate=growth_rate,
            block_config=[depth]*n_blocks
        ).features

        self.n_features = self.cnn.norm5.num_features
    
    def forward(self, x):
        return self.cnn(x)
        
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.attn = nn.Linear(attention_dim, 1)

    def forward(self, decoder_hidden_state, encoder_outputs):
        '''
        encoder_outputs: [T, B, C]
        decoder_hidden_state: [num_layers*num_directions, B, H] = [1, B, H]
        '''
        # pdb.set_trace()
        att1 = self.encoder_att(encoder_outputs) # [T, B, A]
        att2 = self.decoder_att(decoder_hidden_state) # [1, B, A]
        att = self.attn(att1 + att2) # [T, B, 1]
        
        att = att.squeeze(2) # [T, B]
        weights = F.softmax(att, dim=0) # [T, B]
        weights = weights.unsqueeze(2) # [T, B, 1]

        context = weights * encoder_outputs
        context = torch.sum(context, dim=0).unsqueeze(0)

        # context: [1, B, C]
        # weights: [T, B, 1]
        return context, weights
        
class Decoder(nn.Module):
    '''
    Decode from previous character and current input
    '''
    def __init__(self, input_size, vocab_size, hidden_size, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.rnn = nn.GRU(self.input_size + self.vocab_size, self.hidden_size)

    def forward(self, prev_character, current_input, prev_hidden_state):
        '''
        prev_character: [1, B, V]
        current_input:  [1, B, I]
        prev_hidden_state: [num_layers*num_directions, B, H] = [1, B, H]
        '''
        rnn_input = torch.cat([current_input, prev_character], dim=-1) # [1, B, I+V]
#         packed = nn.utils.rnn.pack_padded_sequence(rnn_input, label_lengths)[0]
        return self.rnn(rnn_input, prev_hidden_state)

    def init_hidden(self, device):
        return torch.zeros(1, self.batch_size, self.hidden_size).to(device)

class Model(nn.Module):
    def __init__(self, batch_size, hidden_size, vocab_size, device):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder()
        self.input_size = self.encoder.n_features
        self.attn_size = hidden_size
        self.attn = Attention(self.input_size, hidden_size, self.attn_size)
        self.decoder = Decoder(self.input_size, vocab_size, hidden_size, batch_size)

        self.final = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch_X, batch_Y, batch_Y_lengths):
        '''
        batch_X: [B, C, H, W]
        batch_Y: [T, B, V]
        batch_Y_lengths: [B, 1]
        '''
#         pdb.set_trace()

        encoder_outputs = self.encoder(batch_X) # [B, C, H, W]
        encoder_outputs = encoder_outputs.view(-1, self.batch_size, self.input_size) # [T, B, C], T = H*W

        decoder_hidden = self.decoder.init_hidden(self.device)
        decoder_outputs = []


        for t in range(batch_Y.size(0)):
            y_t = batch_Y[t].unsqueeze(0) # [1, B, V]
            context, weights = self.attn(decoder_hidden, encoder_outputs)
            # context: [1, B, I]
            # weights: [T, B, 1]
            decoder_output, decoder_hidden = self.decoder(y_t, context, decoder_hidden) # teacher forcing

            decoder_output = self.final(decoder_output)
            decoder_outputs.append(decoder_output)

        predict_Y = torch.cat(decoder_outputs, dim=0)
        return predict_Y # [T, B, V]
