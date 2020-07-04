import copy
from argparse import ArgumentParser
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from config import initialize
from metrics import compute_cer, compute_wer
from model import ASPP, STN, PositionalEncoding1d, PositionalEncoding2d
from utils import StringTransform, length_to_padding_mask

from .attention import get_attention
from .model_ce import ModelCE
from .transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = [
    'ModelRNN'
]


class LSTMAttnDecoderLayer(nn.Module):
    def __init__(self, input_size, memory_size, hidden_size, attn_size, attention='additive'):
        super().__init__()
        self.lstm = nn.LSTMCell(
            input_size=input_size + attn_size,
            hidden_size=hidden_size,
        )
        self.Mc = nn.Linear(memory_size, attn_size)
        self.Hc = nn.Linear(hidden_size, attn_size)
        self.attention = get_attention(attention, attn_size)

    def forward(self, input, memory, hidden, cell, output_weights: bool):
        '''
        Shapes:
        -------
        - input: [B,V]
        - memory: [B,S,E]
        - hidden: [B, H]
        - cell: [B, H]
        '''
        attn_hidden = self.Hc(hidden) # [B, A]
        attn_memory = self.Mc(memory) # [B, S, A]
        context, weight = self.attention(attn_hidden.unsqueeze(1),   # [B,1,A]
                                         attn_memory, # [B,S,A]
                                         attn_memory,
                                         output_weights=output_weights) # [B,S,A]
        # context: [B,1,A], weights: [B,1,S]
        lstm_input = torch.cat([input, context.squeeze(1)], dim=-1) # [B, V+A]
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        if output_weights:
            return hidden, cell, weight
        else:
            return hidden, cell, None


class LSTMAttnDecoder(nn.Module):
    def __init__(self, layer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, input, memory, hidden, cell, output_weights: bool):
        weights = []
        for mod in self.layers:
            hidden, cell, weight = mod(input, memory, hidden, cell, output_weights)
            weights.append(weight)
        if output_weights:
            weights = torch.stack(weights, dim=0) # [L,B,T,S]
            return hidden, cell, weights
        else:
            return hidden, cell, None


class ModelRNN(ModelCE):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_length', type=int, default=15)
        parser.add_argument('--beam_width', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--smoothing', type=float, default=0)
        parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
        parser.add_argument('--stn', action='store_true', default=False)
        parser.add_argument('--aspp', action='store_true', default=False)

        lstm_group = parser.add_argument_group()
        lstm_group.add_argument('--attention', type=str, choices=['scaledotproduct', 'additive'], default='additive')
        lstm_group.add_argument('--attn_size', type=int, default=256)
        lstm_group.add_argument('--hidden_size', type=int, default=256)
        lstm_group.add_argument('--num_layers', type=int, default=1)

        tf_enc_group = parser.add_argument_group()
        tf_enc_group.add_argument('--nhead', type=int, default=8)
        tf_enc_group.add_argument('--dim_feedforward', type=int, default=4096)
        tf_enc_group.add_argument('--dropout', type=float, default=0.1)
        tf_enc_group.add_argument('--encoder_nlayers', type=int, default=1)
        tf_enc_group.add_argument('--tf_encoder', action='store_true', default=False)
        return parser


    def __init__(self, config):
        super().__init__(config)
        self.cnn = initialize(config['cnn'])
        self.vocab = initialize(config['vocab'], add_blank=False)

        self.hidden_size = config['hidden_size']
        self.register_buffer('start_index', torch.tensor(self.vocab.SOS_IDX, dtype=torch.long))
        attn_size = config['attn_size']

        lstm_layer = LSTMAttnDecoderLayer(self.vocab.size,
                                          attn_size,#   self.cnn.n_features,
                                          self.hidden_size,
                                          attn_size)
        self.rnn = LSTMAttnDecoder(lstm_layer, config['num_layers'])

        self.Ic = nn.Linear(self.cnn.n_features, attn_size)
        self.Hc = nn.Linear(self.hidden_size, attn_size)
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']

        self.character_distribution = nn.Linear(self.hidden_size, self.vocab.size)

        if config.get('use_stn', False):
            self.stn = STN(in_channels=3)
        else:
            self.stn = nn.Identity()

        if config.get('use_aspp', False):
            self.aspp = ASPP(self.cnn.n_features, self.cnn.n_features)
        else:
            self.aspp = nn.Identity()

        if config.get('tf_encoder', False):
            layer = TransformerEncoderLayer(self.cnn.get_n_features(),
                                            config['nhead'],
                                            dim_feedforward=config['dim_feedforward'],
                                            dropout=config['dropout'])
            self.encoder = TransformerEncoder(layer,
                                              config['encoder_nlayers'],
                                              nn.LayerNorm(self.cnn.get_n_features()))
        else:
            self.encoder = nn.Identity()

        self.string_tf = StringTransform(self.vocab)
        
    def _init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def embed_image(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - image_features: [B,S,E]
        '''
        images = self.stn(images) # [B, C, H, W]
        image_features = self.cnn(images) # [B, C', H', W']
        image_features = self.aspp(image_features) # [B, C', H', W']
        B, C, H, W = image_features.shape
        image_features = image_features.transpose(-2, -1) # [B,C',W',H']
        image_features = image_features.reshape(B, C, W*H) # [B, C', S=W'xH']
        image_features = image_features.permute(2,0,1) # [S, B, C']
        image_features = self.encoder(image_features)  # [S, B, C']
        image_features = image_features.transpose(0, 1) # [B, S, C']
        return image_features

    def embed_text(self, text: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
        - text: [B,T]

        Returns:
        --------
        - text: [B,T,V]
        '''
        text = F.one_hot(text, self.vocab.size).float().to(text.device)
        return text

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        label_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - labels: [B,T]
        - image_padding_mask: [B,H,W]
        - label_padding_mask: [B,T]

        Returns:
        ----
        - outputs: [B,T,V]
        '''
        embedded_image = self.embed_image(images, image_padding_mask) # [B,S,E]
        embed_text = self.embed_text(labels) # [B,T,E]

        batch_size = embedded_image.size(0)
        max_length = embed_text.size(1)
        embedded_image = self.Ic(embedded_image) # [B, S, E]

        rnn_input = embed_text[:, 0].float() # [B,V]
        hidden = self._init_hidden(batch_size).to(embedded_image.device) # [B,H]
        cell = self._init_hidden(batch_size).to(embedded_image.device) # [B,H]

        outputs = torch.zeros(batch_size, max_length, self.vocab.size, device=embedded_image.device)
        for t in range(max_length):
            hidden, cell, _ = self.rnn(input=rnn_input,
                                       memory=embedded_image,
                                       hidden=hidden,
                                       cell=cell,
                                       output_weights=False)
            output = self.character_distribution(hidden) # [B, V]
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            if self.training and teacher_force:
                rnn_input = embed_text[:, t]
            else:
                output = output.argmax(-1, keepdim=True) # [B, 1]
                rnn_input = self.embed_text(output).squeeze(1)

        return outputs

    def greedy(
        self,
        images: torch.Tensor,
        max_length: int,
        image_padding_mask: Optional[torch.Tensor] = None,
        output_weights: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        - weights: weights if `output_weights` is True, else None
        '''
        embedded_image = self.embed_image(images) # [B,S,C']

        num_pixels = embedded_image.size(1)
        batch_size = embedded_image.size(0)

        embedded_image = self.Ic(embedded_image)
        rnn_input = self.embed_text(self.start_index.expand(batch_size).unsqueeze(-1)).squeeze(1) # [B,V]

        hidden = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]
        cell = self._init_hidden(batch_size).to(embedded_image.device) # [B,H]

        outputs = torch.zeros(batch_size, max_length, device=embedded_image.device, dtype=torch.long)

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        weights = []
        for t in range(max_length):
            hidden, cell, weight = self.rnn(input=rnn_input,
                                            memory=embedded_image,
                                            hidden=hidden,
                                            cell=cell,
                                            output_weights=output_weights)
            # hidden: [B,H], cell: [B,H], weights: [L,B,1,S]
            weights.append(weight)
            output = self.character_distribution(hidden) # [B,V]
            output = output.argmax(-1) # [B]
            outputs[:, t] = output
            rnn_input = self.embed_text(output.unsqueeze(-1)).squeeze(1) # [B,V]

            current_end = output.cpu() == self.vocab.char2int(self.vocab.EOS)
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        if output_weights:
            weights = torch.cat(weights, dim=2) # [L,B,T,S]
            return outputs, lengths, weights
        else:
            return outputs, lengths, None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
