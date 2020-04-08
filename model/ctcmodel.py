import torch
import torch.nn as nn
import torch.nn.functional as F

from queue import PriorityQueue
from typing import List, Dict, Tuple
from .positional_encoding import PositionalEncoding1d, PositionalEncoding2d

__all__ = [
    'CTCModel', 'CTCModelTFEncoder', 'CTCModelRNN',
]

class _BeamSearchNode(object):
    def __init__(self,
        prev_chars: List,
        prev_node: '_BeamSearchNode',
        current_char: int,
        log_prob: float,
        length: int
    ):
        self.prev_chars = prev_chars
        self.prev_node = prev_node
        self.current_char = current_char
        self.log_prob = log_prob
        self.length = length

    def eval(self):
        return self.log_prob / float(self.length - 1 + 1e-6)

    def __lt__(self, other):
        return self.length < other.length

    def new(self, char_index: int, log_prob: float):
        new_node = _BeamSearchNode(
            self.prev_chars + [self.current_char],
            self,
            char_index,
            self.log_prob + log_prob,
            self.length + 1,
        )
        return new_node

class CTCModel(nn.Module):
    '''
    Base class for all model use CTC as loss function. This loss expects model return a sequence from
    image so that it can alignment between sequence from image, e.g. (N,S) and sequence from groundtruth, e.g. (N,T)
    '''
    def __init__(self, cnn, vocab, config: Dict):
        super().__init__()
        self.cnn = cnn
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.vocab = vocab

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]

        Returns:
        --------
            - outputs: [B,S,V]
        '''
        images = self.cnn(images) # [B,C,H,W]
        images = self.pool(images) # [B,C,1,W]
        images = images.squeeze(-2).transpose(-1, -2) # [B,S=W,C]
        text = self._forward_decode(images) # [B,S,V]
        return text

    def _forward_decode(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,S,C]

        Returns:
        --------
            - outputs: [B,S,V]
        '''
        pass

    def greedy(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
        Returns:
            - outputs: [B,S]
        '''
        images = self(images) # [B,S,V]
        return images.argmax(-1) # [B,S]

class CTCModelTFEncoder(CTCModel):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab, config)
        # use TransformerEncoderLayer/TransformerEncoder as decoder instead of e.g. LSTM
        decoder_layer = nn.TransformerEncoderLayer(d_model=cnn.n_features, nhead=config['nhead'])
        self.decoder = nn.TransformerEncoder(decoder_layer, config['num_layers'])
        self.character_distribution = nn.Linear(cnn.n_features, vocab.size) # [B,S,V]

    def _forward_decode(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            images: (N,S,C)
            outputs: (N,S,C)
        '''
        images = images.transpose(0,1) # [S,B,C]
        images = self.decoder(images) # [S,B,C]
        images = images.transpose(0,1) # [B,S,C]
        outputs = self.character_distribution(images) # [B,S,V]
        return outputs

class CTCModelRNN(CTCModel):
    def __init__(self, cnn, vocab, config):
        super().__init__(cnn, vocab, config)

        self._num_layers = config['num_layers']
        self._num_direct = 2 if config['bidirectional'] else 1
        self._hidden_size = config['hidden_size']

        self.decoder = nn.LSTM(self.cnn.n_features,
                               config['hidden_size'],
                               config['num_layers'],
                               batch_first=True,
                               dropout=config['dropout'],
                               bidirectional=config['bidirectional'])

        self.character_distribution = nn.Linear(self._num_direct * self._hidden_size, vocab.size)

    def _init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(self._num_direct*self._num_layers, batch_size, self._hidden_size)
        cell = torch.zeros(self._num_direct*self._num_layers, batch_size, self._hidden_size)
        return (hidden, cell)

    def _forward_decode(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            images: (N,S,C)
            outputs: (N,S,C)
        '''
        hidden_init, cell_init = self._init_hidden(images.size(0))
        hidden_init, cell_init = hidden_init.to(images.device), cell_init.to(images.device)
        outputs, _ = self.decoder(images, (hidden_init, cell_init)) # [B,S,H]
        outputs = self.character_distribution(outputs) # [B,S,V]
        return outputs
