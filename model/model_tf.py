from argparse import ArgumentParser
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from config import initialize
from metrics import compute_cer, compute_wer
from utils import StringTransform, length_to_padding_mask

from .aspp import ASPP
from .feature_extractor import ResnetFE
from .model_ce import ModelCE
from .positional_encoding import PositionalEncoding1d, PositionalEncoding2d
from .stn import STN
from .transformer import (Transformer, TransformerDecoder,
                          TransformerDecoderLayer, TransformerEncoder,
                          TransformerEncoderLayer,
                          generate_square_subsequent_mask)

__all__ = ['ModelTF']


class _BeamSearchNode(object):
    def __init__(self,
        prev_prob: List[torch.Tensor], # [ [V] ]
        prev_node: '_BeamSearchNode',
        log_prob: float,
        current_char: torch.Tensor, # [V]
    ):
        self.prev_prob = prev_prob
        self.prev_node = prev_node
        self.current_char = current_char
        self.log_prob = log_prob

    def eval(self):
        return self.log_prob / float(len(self.prev_prob) - 1 + 1e-6)

    def __lt__(self, other):
        return len(self.prev_prob) < len(other.prev_prob)

    def new(self, current_char: torch.Tensor, log_prob: float):
        '''
        Shapes:
        -------
        - current_char: [V]
        '''
        new_node = _BeamSearchNode(
            self.prev_prob + [self.current_char],
            self,
            self.log_prob + log_prob,
            current_char,
        )
        return new_node


class ModelTF(ModelCE):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_length', type=int, default=15)
        parser.add_argument('--beam_width', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--smoothing', type=float, default=0)
        parser.add_argument('--attn_size', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dim_feedforward', type=int, default=4096)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--encoder_nlayers', type=int, default=0)
        parser.add_argument('--decoder_nlayers', type=int, default=1)
        parser.add_argument('--stn', action='store_true', default=False)
        parser.add_argument('--aspp', action='store_true', default=False)
        parser.add_argument('--pe_text', action='store_true', default=False)
        parser.add_argument('--pe_image', action='store_true', default=False)
        return parser

    def __init__(self, config):
        super().__init__(config)
        self.cnn = initialize(config['cnn'])
        self.vocab = initialize(config['vocab'], add_blank=False)
        self.register_buffer('start_index', torch.tensor(self.vocab.SOS_IDX, dtype=torch.long))

        self.Ic = nn.Linear(self.cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(self.vocab.size, config['attn_size'])
        self.character_distribution = nn.Linear(config['attn_size'], self.vocab.size)

        decoder_layer = TransformerDecoderLayer(config['attn_size'],
                                                config['nhead'],
                                                dim_feedforward=config['dim_feedforward'],
                                                dropout=config['dropout'])
        decoder_norm = nn.LayerNorm(config['attn_size'])
        self.decoder = TransformerDecoder(decoder_layer, config['decoder_nlayers'], decoder_norm)

        if config.get('encoder_nlayers', 0) > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=config['attn_size'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
            )
            encoder_norm = nn.LayerNorm(config['attn_size'])
            self.encoder = TransformerEncoder(encoder_layer, config['encoder_nlayers'], encoder_norm)
        else:
            self.encoder = nn.Identity()

        if config.get('stn', False):
            self.stn = STN(in_channels=3)
        else:
            self.stn = nn.Identity()

        if config.get('aspp', False):
            self.aspp = ASPP(self.cnn.n_features, self.cnn.n_features)
        else:
            self.aspp = nn.Identity()

        if config.get('pe_text', False):
            self.pe_text = PositionalEncoding1d(config['attn_size'], batch_first=True)
        else:
            self.pe_text = nn.Identity()

        if config.get('pe_image', False):
            self.pe_image = PositionalEncoding2d(config['attn_size'])
        else:
            self.pe_image = nn.Identity()

        self.string_tf = StringTransform(self.vocab)

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
        - image_features: [B,E,H,W]
        '''
        image_features = self.stn(images) # [B,C',H',W']
        image_features = self.cnn(image_features) # [B, C', H', W']
        image_features = self.aspp(image_features) # [B, C', H', W']
        image_features = image_features.permute(0,2,3,1) # [B, H', W', C']
        image_features = self.Ic(image_features) # [B,H',W',E]
        image_features = image_features.permute(0,3,1,2) # [B, E, H', W']
        image_features = self.pe_image(image_features) # [B,E,H',W']
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
        embed_image = self.embed_image(images, image_padding_mask) # [B,E,H,W]

        B, E, H, W = embed_image.shape
        embed_image = embed_image.transpose(-2, -1) # [B,E,W',H']
        embed_image = embed_image.reshape(B, E, W*H) # [B, E, S=W'xH']
        embed_image = embed_image.permute(2,0,1) # [S,B,E]
        embed_image = self.encoder(embed_image) # [S,B,E]
        if isinstance(embed_image, tuple):
            embed_image, _ = embed_image

        embed_text = self.embed_text(labels) # [B,T,V]
        embed_text = self.Vc(embed_text) # [B,T,E]
        embed_text = self.pe_text(embed_text) # [B,T,E]
        embed_text = embed_text.transpose(0, 1) # [T,B,E]

        attn_mask = generate_square_subsequent_mask(embed_text.size(0)).to(embed_text.device)
        outputs, _ = self.decoder(embed_text, embed_image,
                                  tgt_mask=attn_mask,
                                  tgt_key_padding_mask=label_padding_mask)
        outputs = outputs.transpose(0, 1)
        outputs = self.character_distribution(outputs)
        return outputs

    def decode(
        self,
        images: torch.Tensor,
        max_length: int,
        beam_width: int,
        image_padding_mask: Optional[torch.Tensor] = None,
        output_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        - weights: (Optional[[B,H',W',H',W']], Optional[[B,T,H',W']], Optional[[B,T,T]] or None if output_weights=False
        '''
        # if beam_width > 1:
        #     return self.beamsearch(images, max_length, beam_width, image_padding_mask=image_padding_mask)
        # else:
        #     return self.greedy(images, max_length, image_padding_mask=image_padding_mask)
        return self.greedy(images, max_length, image_padding_mask=image_padding_mask, output_weights=output_weights)

    def greedy(
        self,
        images: torch.Tensor,
        max_length: int,
        image_padding_mask: Optional[torch.Tensor] = None,
        output_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        - weights: (Optional[[B,H',W',H',W']], Optional[[B,T,H',W']], Optional[[B,T,T]] or None if output_weights=False
        '''
        batch_size = len(images)
        images = self.embed_image(images, image_padding_mask) # [B,E,H,W]

        B, E, H, W = images.shape
        images = images.transpose(-2, -1) # [B,E,W,H]
        images = images.reshape(B, E, W*H) # [B, E, S=W*H]
        images = images.permute(2,0,1) # [S,B,E]
        images = self.encoder(images) # [S,B,E]
        if isinstance(images, tuple):
            images, enc_weights = images
        else:
            enc_weights = None
        # TODO: reshape weights for encoder

        images = images.transpose(0,1) # [B,S,E]

        predicts = self.start_index.expand(batch_size).unsqueeze(-1) # [B,1]
        predicts = self.embed_text(predicts) # [B,1,V]

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            output, (self_attn_w, attn_w) = self.inference_step(images, predicts)
            # [B,V], [L,B,T,T], [L,B,T,S]
            output = F.softmax(output, dim=-1) # [B,V]
            predicts = torch.cat([predicts, output.unsqueeze(1)], dim=1) # [B,T,V]

            output = output.argmax(-1) # [B]
            current_end = output.cpu() == self.vocab.char2int(self.vocab.EOS)
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        predicts = predicts[:, 1:].argmax(-1)
        if output_weights:
            attn_w = attn_w.reshape(-1, B, t+1, W, H).transpose(-2, -1)  # [L,B,T,H,W]
            return predicts, lengths, (enc_weights, attn_w, self_attn_w)
        else:
            return predicts, lengths, None

    def inference_step(
        self,
        embedded_image: torch.Tensor,
        predicts: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Shapes:
        -------
        - embedded_image: [B,S,E]
        - predicts: [B,T,V]

        Returns:
        --------
        - logits: [B,V]
        - self_attn_weights, attn_weights: [L,B,T,T], [L,B,T,S]
        '''
        text = self.Vc(predicts) # [B,T,E]
        text = self.pe_text(text) # [B,T,E]
        text = text.transpose(0,1) # [T,B,E]
        attn_mask = generate_square_subsequent_mask(text.size(0)).to(text.device)
        embedded_image = embedded_image.transpose(0, 1) # [S,B,E]
        output, (self_attn_w, attn_w) = self.decoder(text, embedded_image, tgt_mask=attn_mask)
        # output: [T,B,E], self_attn_w: [L,B,T,T], attn_w: [L,B,T,S]
        output = output.transpose_(0, 1) # [B,T,E]
        logits = self.character_distribution(output[:,-1]) # [B,V]
        return logits, (self_attn_w, attn_w)

    # def beamsearch(self, images: torch.Tensor):
    #     '''
    #     Shapes:
    #     -------
    #     - images: [B,C,H,W]
    #     Returns:
    #     --------
    #     - outputs: [B,T]
    #     '''

    #     def decode_one_sample(image: torch.Tensor, start: torch.Tensor, max_length: int, beam_width: int):
    #         '''
    #         image: [S,E]
    #         start: [V]
    #         '''
    #         node = _BeamSearchNode([], None, 0, start)
    #         nodes = []
    #         endnodes = []

    #         # start the queue
    #         nodes = PriorityQueue()
    #         nodes.put_nowait((-node.eval(), node))

    #         # start beam search
    #         image = image.unsqueeze(0) # [B=1,S,E]
    #         while True:
    #             # give up when decoding takes too long
    #             if nodes.qsize() > 2000: break

    #             # fetch the best node
    #             score: float
    #             node: _BeamSearchNode
    #             score, node = nodes.get()
    #             if node.current_char.argmax(-1) == self.vocab.char2int(self.vocab.EOS) and node.prev_node is not None:
    #                 endnodes.append((score, node))
    #                 # if we reached maximum # of sentences required
    #                 if len(endnodes) >= beam_width:
    #                     break
    #                 else:
    #                     continue

    #             # decode for one step using decoder
    #             predicts = torch.stack(node.prev_prob + [node.current_char], dim=0) # [T,V]
    #             predicts = predicts.unsqueeze(0).to(image.device) # [B=1,T,V]
    #             logits = self.inference_step(image, predicts) # [B=1,V]
    #             log_prob = F.log_softmax(logits, dim=-1).squeeze(0) # [V]
    #             prob = F.softmax(logits, dim=-1).squeeze(0) # [V]

    #             # PUT HERE REAL BEAM SEARCH OF TOP
    #             log_probs, indexes = log_prob.topk(beam_width, -1)
    #             for log_prob in log_probs.cpu().tolist():
    #                 new_node = node.new(prob, log_prob)
    #                 nodes.put_nowait((-new_node.eval(), new_node))

    #         # choose nbest paths, back trace them
    #         if len(endnodes) == 0:
    #             endnodes = [nodes.get_nowait() for _ in range(beam_width)]

    #         # Only get the maximum prob
    #         # print([x[0] for x in endnodes])
    #         score, node = sorted(endnodes, key=lambda x: x[0])[0]
    #         # print(f'Best score: {-score}, length = {node.length}')
    #         s = [node.current_char.argmax(-1)]
    #         # back trace
    #         while node.prev_node is not None:
    #             node = node.prev_node
    #             s.insert(0, node.current_char.argmax(-1))
    #         return torch.tensor(s, dtype=torch.long)

    #     batch_size = len(images)
    #     images = self.embed_image(images) # [B,S,E]
    #     images.transpose_(0, 1) # [S,B,E]

    #     starts = self.embed_text(self.start_index.expand(batch_size, 1)).squeeze(1) # [B,V]
    #     decoded_batch = []
    #     lengths = []

    #     # decoding goes sentence by sentence
    #     for idx in range(batch_size):
    #         string_index = decode_one_sample(images[:, idx], starts[idx], self.max_length, self.beam_width)
    #         decoded_batch.append(string_index)
    #         lengths.append(len(string_index) - 1) # ignore SOS

    #     decoded_batch = torch.nn.utils.rnn.pad_sequence(decoded_batch, batch_first=True)[:, 1:]
    #     lengths = torch.tensor(lengths, dtype=torch.long)

    #     return decoded_batch, lengths
