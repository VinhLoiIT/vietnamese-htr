import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random

from .attention import Attention, MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, cnn, vocab_size, attn_size, encoder_nhead, decoder_nhead, both_nhead, encoder_nlayers=1, decoder_nlayers=1):
        super().__init__()

        self.cnn = cnn
        self.vocab_size = vocab_size

        encoder_layer = TransformerEncoderLayer(self.cnn.n_features, nhead=encoder_nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_nlayers)

        decoder_layer = TransformerDecoderLayer(self.cnn.n_features, vocab_size, attn_size, nhead=decoder_nhead)
        self.decoder = TransformerDecoder(attn_size, decoder_layer, num_layers=decoder_nlayers)

        self.character_distribution = nn.Linear(attn_size, vocab_size)

    def forward(self, images, targets, teacher_forcing_ratio=0.5):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param targets: Tensor of [L,B,V], which should start with <start> and end with <end>
        Return:
            - outputs: [L,B,V]
        '''
        max_length = targets.size(0) - 1 # ignore <start>
        batch_size, _, input_image_h, input_image_w = images.size()

        # Step 1: CNN Feature Extraction
        image_features = self.cnn(images) # [B, C', H', W']
        feature_image_h, feature_image_w = image_features.size()[-2:]
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.permute(2,0,1) # [S, B, C']

        # Step 2: Encoder forwarding
        image_features, _ = self.encoder.forward(image_features, output_weights=False)

        # Step 3: Decoder forwarding
        targets = targets.float()
        # predicts = torch.zeros(max_length, batch_size, self.vocab_size, dtype=torch.float32, device=targets.device)
        # predicts[[0]] = targets[[0]]
        predicts = targets[[0]]

        for t in range(max_length):
            # output, _ = self.decoder.forward(predicts, image_features)
            # output = self.character_distribution(output)
            # predicts = torch.cat([predicts, output], dim=0)
            output, _ = self.decoder.forward(targets[:t+1], image_features)
            output = self.character_distribution(output)
            predicts = torch.cat([predicts, output], dim=0)

        return predicts

    def greedy(self, images, start_input, output_weight=False, max_length=10):
        '''
        Inputs:
        :param images: [B,C,H,W]
        :param start_input: Tensor of [L,B,V], which should start with <start> and end with <end>
        Return:
            - outputs: [L,B,V]
            - weights: None #TODO: not implement yet
        '''
        batch_size, _, input_image_h, input_image_w = images.size()

        # Step 1: CNN Feature Extraction
        image_features = self.cnn(images) # [B, C', H', W']
        feature_image_h, feature_image_w = image_features.size()[-2:]
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.permute(2,0,1) # [S, B, C']

        # Step 2: Encoder forwarding
        image_features, _ = self.encoder.forward(image_features, output_weights=False)

        # Step 3: Decoder forwarding
        predicts = start_input.float()
        weights = []
        for t in range(max_length):
            output, weight = self.decoder.forward(predicts, image_features)
            output = self.character_distribution(output)
            output = F.softmax(output, -1)
            index = output.topk(1, -1)[1]
            output = torch.zeros_like(output)
            output.scatter_(-1, index, 1)
            predicts = torch.cat([predicts, output], dim=0)
            weights.append(weight)

        return predicts, weights

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

class TransformerDecoder(nn.Module):
    def __init__(self, attn_size, decoder_layer, num_layers=1):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        # self.norm = nn.LayerNorm(attn_size)

    def forward(self, tgt, image_features, output_weight=False):

        output = tgt
        weights = []
        for i in range(self.num_layers):
            output, weight = self.layers[i](image_features, tgt)
            weights.append(weight)
        # output = self.norm(output)

        if output_weight:
            return output, weights
        else:
            return output, None

class TransformerDecoderLayer(nn.Module):
    def __init__(self, cnn_features, vocab_size, attn_size, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(vocab_size, vocab_size, vocab_size)
        self.encoder_decoder_attn = MultiHeadAttention(cnn_features, vocab_size, attn_size)

        # self.self_attn = nn.modules.MultiheadAttention(attn_size, nhead, dropout=dropout)
        # self.multihead_attn = nn.modules.MultiheadAttention(attn_size, nhead, dropout=dropout)

        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(attn_size, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, attn_size)

        # self.norm1 = nn.LayerNorm(vocab_size)
        # self.norm2 = nn.LayerNorm(attn_size)
        # self.norm3 = nn.LayerNorm(attn_size)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.attn_size = attn_size

    def forward(self, image_features, tgt, output_weights=False):
        context_text, weight_text = self.self_attn.forward(tgt[[-1]], tgt)
        # context_text = tgt + self.dropout1(context_text)
        # context_text = self.norm1(context_text)

        context_attn, weight_attn = self.encoder_decoder_attn.forward(context_text, image_features)
        # context_attn = context_attn + self.dropout2(context_attn)
        # context_attn = self.norm2(context_attn)

        # tgt2 = self.linear2(self.dropout(F.relu(self.linear1(context_img))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        # return tgt, weight

        if output_weights:
            return context_attn, (weight_text, weight_attn)
        else:
            return context_attn, None

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # self.norm = nn.LayerNorm(encoder_layer.feature_size)

    def forward(self, img_features, output_weights=False):
        output = img_features

        weights = []
        for i in range(self.num_layers):
            output, weight = self.layers[i](output, output_weights)
            weights.append(weights)

        # output = self.norm(output)

        if output_weights:
            return output, weights
        else:
            return output, None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_size, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.feature_size = feature_size
        self.self_attn = MultiHeadAttention(feature_size, feature_size, feature_size, nhead=nhead)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(feature_size, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, feature_size)

        # self.norm1 = nn.LayerNorm(feature_size)
        # self.norm2 = nn.LayerNorm(feature_size)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, img_features, output_weights=False):
        img_features, weight = self.self_attn(img_features, img_features)
        # img_features = img_features + self.dropout1(img_features2)
        # img_features = self.norm1(img_features)
        # img_features2 = self.linear2(self.dropout(F.relu(self.linear1(img_features))))
        # img_features = img_features + self.dropout2(img_features2)
        # img_features = self.norm2(img_features)
        if output_weights:
            return img_features, weight
        else:
            return img_features, None



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# if __name__ == "__main__":
#     attn_size = 256
#     vocab_size = 100
#     feature_size = 200
#     batch_size = 2
#     model = TransformerModel(feature_size, vocab_size, attn_size)
#     print(model)
#     with torch.no_grad():
#         tgt = torch.ones(5, batch_size, vocab_size)
#         image_features = torch.rand(150, batch_size, feature_size)
#         outputs = model.forward(image_features, tgt, None, None, None)
#         outputs = F.softmax(outputs, -1)
#     print('outputs:', outputs.size())
#     print(outputs)
