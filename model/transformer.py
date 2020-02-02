import torch
import torch.nn as nn
import torch.nn.functional as F

import math
# from .encoder import Encoder


class TransformerModel(nn.Module):
    def __init__(self, cnn_features, vocab_size, d_model):
        super(TransformerModel, self).__init__()
        decoder_layer = nn.modules.TransformerDecoderLayer(d_model, 8)
        self.transformer = nn.modules.TransformerDecoder(decoder_layer, num_layers=6, norm=None)
        
        self.linear_img = nn.Linear(cnn_features, d_model)
        self.linear_text = nn.Linear(vocab_size, d_model)
        
        self.linear_output = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        
        self.positional_encoding_text = PositionalEncoding(d_model)
    
    def forward(self, img_features, target_onehots, targets, targets_length, PAD_CHAR_int):
        img_features = self.linear_img(img_features)
        # Optional: PositionEncoding for imgs
        # here ...
        
        targets_onehot = self.linear_text(target_onehots.float())
        targets_onehot = self.positional_encoding_text(targets_onehot)
        
        tgt_mask = nn.modules.Transformer.generate_square_subsequent_mask(None, torch.max(targets_length)).to(targets.device) # ignore SOS_CHAR
        tgt_key_padding_mask = (targets == PAD_CHAR_int).squeeze(-1).transpose(0,1).to(targets.device) # [B,T]
        
        memory_mask = None
        memory_key_padding_mask = None
        outputs = self.transformer.forward(targets_onehot, img_features,
                                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask, 
                                           memory_key_padding_mask=memory_key_padding_mask)
        # [T, N, d_model]
        outputs = self.linear_output(outputs) # [T, N, vocab_size]
        return outputs
    
    def inference(self, img_features, start_input, max_length=10):
        img_features = self.linear_img(img_features)
        # Optional: PositionEncoding for imgs
        # here ...
        
        outputs = start_input
        
        for t in range(max_length):
            transformer_input = self.linear_text(outputs)
            transformer_input = self.positional_encoding_text(transformer_input)
            output = self.transformer.forward(transformer_input, img_features,
                                              tgt_mask=None, memory_mask=None,
                                              tgt_key_padding_mask=None,
                                              memory_key_padding_mask=None)
            output = self.linear_output(output[[-1]])
            output = F.softmax(output, -1)
            _, index = output.topk(1, -1)
            predict = torch.zeros(1, 1, self.vocab_size).to(img_features.device)
            predict[:,:,index] = 1
            outputs = torch.cat([outputs, predict], dim=0)

        return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)