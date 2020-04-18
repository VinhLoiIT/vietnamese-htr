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
    
class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.p_total = 0 # blank + non-blank
        self.p_nonblank = 0 # non-blank
        self.p_blank = 0 # blank
        self.labeling = () # beam-labeling

class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}
        
    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.p_total)
        return [x.labeling for x in sorted_beams]
        
def beamsearch_ctc(output_ctc, beam_width=3):
    max_len, vocab_size = output_ctc.shape
    blank_idx = vocab_size - 1

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].p_blank = 1
    last.entries[labeling].p_total = 1

    # go over all time-steps
    for t in range(max_len):
        curr = BeamState()

        # get beam-labelings of best beams
        prev_tops = last.sort()[0:beam_width]

        # go over best beams
        for token in prev_tops:

            # probability of paths ending with a non-blank
            p_nonblank = 0
            # in case of non-empty beam
            if token:
                # probability of paths with repeated last char at the end
                p_nonblank = last.entries[token].p_nonblank * output_ctc[t, token[-1]]

            # probability of paths ending with a blank
            p_blank = (last.entries[token].p_total) * output_ctc[t, blank_idx]

            # add beam at current time-step if needed
            if token not in curr.entries:
                curr.entries[token] = BeamEntry()

            # fill in data
            curr.entries[token].labeling = token
            curr.entries[token].p_nonblank += p_nonblank
            curr.entries[token].p_blank += p_blank
            curr.entries[token].p_total += p_blank + p_nonblank

            # extend current beam-labeling
            for c in range(vocab_size-1):
                # add new char to current beam-labeling
                extend_token = token + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if token and token[-1] == c:
                    p_nonblank = output_ctc[t, c] * last.entries[token].p_blank
                else:
                    p_nonblank = output_ctc[t, c] * last.entries[token].p_total

                # add beam at current time-step if needed
                if extend_token not in curr.entries:
                    curr.entries[extend_token] = BeamEntry()
                
                # fill in data
                curr.entries[extend_token].labeling = extend_token
                curr.entries[extend_token].p_nonblank += p_nonblank
                curr.entries[extend_token].p_total += p_nonblank

        # set new beam state
        last = curr

     # sort by probability
    result = last.sort()[0] # get most probable labeling
    return list(result)

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
        outputs = self(images) # [B,S,V]
        outputs = F.softmax(outputs, -1)
        return outputs.argmax(-1) # [B,S]
    
    def beamsearch(self, images: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
        Returns:
            - outputs: [B,S]
        '''
        outputs = self(images) # [B,S,V]
        outputs = F.softmax(outputs, -1)
        results = []
        for output in outputs:
            result = beamsearch_ctc(output)
            result = [i for i in result]
            result += [self.vocab.char2int(self.vocab.EOS)]
            results.append(torch.tensor(result))
        return torch.nn.utils.rnn.pad_sequence(results, batch_first=True)

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
