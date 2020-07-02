from argparse import ArgumentParser
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from metrics import compute_cer, compute_wer
from model import ASPP, STN, PositionalEncoding1d, PositionalEncoding2d
from utils import StringTransform, length_to_padding_mask
from config import initialize

from .model_ce import ModelCE
from .attention import get_attention

__all__ = [
    'ModelRNN'
]


class ModelRNN(ModelCE):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--attention', type=str, choices=['scaledotproduct', 'additive'], default='additive')
        parser.add_argument('--max-length', type=int, default=15)
        parser.add_argument('--beam-width', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=16)
        parser.add_argument('--smoothing', type=float, default=0)
        parser.add_argument('--attn-size', type=int, default=512)
        parser.add_argument('--hidden_size', type=int, default=8)
        parser.add_argument('--teacher_forcing_ratio', type=float, default=0.1)
        parser.add_argument('--stn', action='store_true', default=False)
        parser.add_argument('--aspp', action='store_true', default=False)
        return parser


    def __init__(self, config):
        super().__init__(config)
        self.cnn = initialize(config['cnn'])
        self.vocab = initialize(config['vocab'], add_blank=False)

        self.hidden_size = config['hidden_size']
        self.register_buffer('start_index', torch.tensor(self.vocab.SOS_IDX, dtype=torch.long))
        self.max_length = config['max_length']
        attn_size = config['attn_size']

        self.rnn = nn.LSTMCell(
            input_size=self.vocab.size+attn_size,
            hidden_size=self.hidden_size,
        )

        self.Ic = nn.Linear(self.cnn.n_features, attn_size)
        self.Hc = nn.Linear(self.hidden_size, attn_size)
        self.attention = get_attention(config['attention'], attn_size)
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
        image_features = self.cnn(images) # [B, C', H', W']
        B, C, H, W = image_features.shape
        image_features = image_features.transpose(-2, -1) # [B,C',W',H']
        image_features = image_features.reshape(B, C, W*H) # [B, C', S=W'xH']
        image_features = image_features.transpose(1,2) # [B, S, C']
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
        embed_image = self.embed_image(images, image_padding_mask) # [B,S,E]
        embed_text = self.embed_text(labels) # [B,T,E]

        batch_size = embed_image.size(0)
        max_length = embed_text.size(1)
        embed_image = self.Ic(embed_image) # [B, S, E]

        rnn_input = embed_text[:, 0].float() # [B,V]
        hidden = self._init_hidden(batch_size).to(embed_image.device) # [B,H]
        cell_state = self._init_hidden(batch_size).to(embed_image.device) # [B,H]

        outputs = torch.zeros(batch_size, max_length, self.vocab.size, device=embed_image.device)
        for t in range(max_length):
            attn_hidden = self.Hc(hidden) # [B, E]
            context, _ = self.attention(attn_hidden.unsqueeze(1), embed_image, embed_image) # [B, 1, attn_size], [B, 1, S]
            context = context.squeeze_(1) # [B, attn_size]
            # self.rnn.flatten_parameters()
            hidden, cell_state = self.rnn(torch.cat((rnn_input, context), -1), (hidden, cell_state))
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
        cell_state = self._init_hidden(batch_size).to(embedded_image.device) # [B, H]

        outputs = torch.zeros(batch_size, self.max_length, device=embedded_image.device, dtype=torch.long)

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        weights = []
        for t in range(self.max_length):
            attn_hidden = self.Hc(hidden) # [B, A]
            context, weight = self.attention(attn_hidden.unsqueeze(1), embedded_image, embedded_image, output_weights=output_weights) # [B, 1, A]
            context.squeeze_(1) #
            weights.append(weight)
            rnn_input = torch.cat((rnn_input, context), -1) # [B, V+A]

            hidden, cell_state = self.rnn(rnn_input, (hidden, cell_state))
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
            weights = torch.cat(weights, dim=1) # [B,T,S]
            return outputs, lengths, weights
        else:
            return outputs, lengths, None
