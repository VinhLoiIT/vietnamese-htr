import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random

from .attention import *

class Transformer(nn.Module):
    def __init__(self, cnn, vocab, config):
        super().__init__()

        self.cnn = cnn
        self.vocab = vocab
        self.Ic = nn.Linear(self.cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(self.vocab.size, config['attn_size'])
        self.character_distribution = nn.Linear(config['attn_size'], self.vocab.size)

        self.transformer = nn.Transformer(
                d_model=config['attn_size'],
                nhead=config['nhead'],
                num_encoder_layers=config['encoder_nlayers'],
                num_decoder_layers=config['decoder_nlayers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
        )
        
        #self.pe_text = PositionalEncoding1d(config['attn_size'], batch_first=True)
        #self.pe_image = PositionalEncoding2d(self.cnn.n_features)

    def embed_image(self, images):
        '''
        Shapes:
        -------
        images: [B,C,H,W]

        Returns:
        --------
        image_features: [B,S,A]
        '''
        image_features = self.cnn(images) # [B, C', H', W']
        batch_size, height, width = images.size(0), images.size(2), images.size(3)
        #image_features = self.pe_image(image_features) # [B,C',H',W']
        image_features = image_features.transpose(-2, -1) # [B,C',W',H']
        image_features = image_features.reshape(batch_size, self.cnn.n_features, -1) # [B, C', S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, C']
        image_features = self.Ic(image_features) # [B,S,A]

        return image_features

    def embed_text(self, text):
        '''
        Shapes:
        -------
        text: [B, T]

        Returns:
        --------
        outputs: [B,T,A]
        '''
        text = F.one_hot(text, self.vocab.size).float().to(text.device)
        text = self.Vc(text) # [B,T,A]
        #text = self.pe_text(text) # [B,T,A]
        return text

    def forward(self, images, targets):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param targets: Tensor of [B,T], which should start with <start> and end with <end>
        Return:
            - outputs: [B,T,V]
        '''
        image_features = self.embed_image(images) # [B, S, A]
        image_features = image_features.transpose_(0, 1) # [S, B, A]

        targets = self.embed_text(targets) # [B, T, A]
        targets = targets.transpose_(0,1) # [T, B, A]

        max_length = targets.size(0)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, max_length).to(targets.device)
        output = self.transformer(image_features, targets, tgt_mask=attn_mask)
        output = self.character_distribution(output.transpose_(0,1))
        return output

    def greedy(self, images, start_input, output_weights=False, max_length=10):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param start_input: Tensor of [B,1], which is <start> character in onehot
        Return:
            - outputs: [B,L,V]
            - weights: None #TODO: not implement yet
        '''
        batch_size = len(images)
        image_features = self.embed_image(images) # [B, S, A]
        image_features = image_features.transpose_(0, 1) # [S, B, A]

        predicts = start_input
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, max_length).to(predicts.device)
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for t in range(max_length):
            targets = self.embed_text(predicts) # [B,T,A]
            targets = targets.transpose_(0,1) # [T,B,A]
            output = self.transformer(image_features, targets, tgt_mask=attn_mask[:t+1, :t+1]) # [T, B, A]
            output = output.transpose_(0, 1) # [B,T,A]
            output = self.character_distribution(output[:,[-1]]) # [B,1,V]
            output = output.argmax(-1) # [B, 1]
            predicts = torch.cat([predicts, output], dim=1)
            
            end_flag |= (output.cpu().squeeze(-1) == self.vocab.char2int(self.vocab.EOS))
            if end_flag.all():
                break
        return predicts[:,1:], None #TODO: return weight

class PositionalEncoding1d(nn.Module):

    def __init__(self, d_model, batch_first=False, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1) # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(1)

        self.batch_first = batch_first
        if self.batch_first:
            pe.transpose_(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding2d(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1) # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [B,C,H,W]
        '''
        x = x.permute(0,2,3,1) # [B,H,W,C]
        xshape = x.shape
        x = x.reshape(-1, x.size(2), x.size(3)) # [B*H,W,C]
        x = x + self.pe[:, :x.size(1), :]
        x = x.reshape(*xshape) # [B,H,W,C]
        x = x.permute(0,3,1,2) # [B,C,H,W]
        return self.dropout(x)
