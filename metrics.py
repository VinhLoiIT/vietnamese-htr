import torch
from torch.nn.utils.rnn import pad_packed_sequence

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics import RunningAverage
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import editdistance as ed

__all__ = [
            'CharacterErrorRate',
            'WordErrorRate',
            ]

def _calc_length(tensor, EOS_int, batch_first=False):
    if not batch_first:
        tensor = tensor.transpose(0,1)

    lengths = []
    for sample in tensor.tolist():
        try:
            end = sample.index(EOS_int)
        except:
            end = None
        lengths.append(end)
    return lengths


class CharacterErrorRate(Metric):
    '''
    Calculates the CharacterErrorRate.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, vocab, output_transform=None, batch_first=True):
        super().__init__()
        self.EOS_int = vocab.char2int(vocab.EOS)
        self.output_transform = output_transform
        self.batch_first = batch_first

    def cer(self, pred, target):
        '''
        Parameters:
        -----------
        pred: [B,T_1]
        target: [B,T_2]

        Output:
        float cer
        '''
        assert len(pred) == len(target)
        batch_size = len(pred)
        pred_lengths = _calc_length(pred, self.EOS_int, True)
        target_lengths = _calc_length(target, self.EOS_int, True)

        batch_cers = 0
        for i, (pred_length, tgt_length) in enumerate(zip(pred_lengths, target_lengths)):
            distance = ed.distance(pred[i, :pred_length].tolist(), target[i, :tgt_length].tolist())
            distance = distance / len(target[i, :tgt_length])
            batch_cers += distance
        return batch_cers

    @reinit__is_reduced
    def reset(self) -> None:
        self._cer = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        if self.output_transform is not None:
            output = self.output_transform(output)
        y_pred, y = output
        y_pred = y_pred.squeeze(-1)
        y = y.squeeze(-1)
        if not self.batch_first:
            y_pred = y_pred.transpose(0,1)
            y = y.transpose(0,1)
        batch_size = len(y_pred)

        self._cer += self.cer(y_pred, y)
        self._num_examples += batch_size

    @sync_all_reduce("_cer", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CharacterErrorRate must have at least one example before it can be computed.')
        return self._cer / self._num_examples

class WordErrorRate(Metric):
    '''
    Calculates the WordErrorRate.
    Notes:
    - When recognize at word-level, this metric is (1 - Accuracy)
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, vocab, output_transform=None, batch_first=True):
        super().__init__()
        self.EOS_int = vocab.char2int(vocab.EOS)
        self.output_transform = output_transform
        self.batch_first = batch_first

    def wer(self, pred, target):
        '''
        Parameters:
        -----------
        pred: [B,T_1]
        target: [B,T_2]

        Output:
        float wer
        '''
        assert len(pred) == len(target), "Batch size must match. pred = {}, target = {}".format(len(pred), len(target))
        batch_size = len(pred)
        pred_lengths = _calc_length(pred, self.EOS_int, True)
        target_lengths = _calc_length(target, self.EOS_int, True)

        batch_wers = 0
        for i, (pred_length, tgt_length) in enumerate(zip(pred_lengths, target_lengths)):
            distance = 0 if torch.equal(pred[i, :pred_length], target[i, :tgt_length]) else 1
            batch_wers += distance
        return batch_wers

    @reinit__is_reduced
    def reset(self) -> None:
        self._wer = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        if self.output_transform is not None:
            output = self.output_transform(output)
        y_pred, y = output
        y_pred = y_pred.squeeze(-1)
        y = y.squeeze(-1)

        if not self.batch_first:
            y_pred = y_pred.transpose(0,1) # [B,T_1]
            y = y.transpose(0,1) # [B,T_2]
        batch_size = y_pred.size(0)

        self._wer += self.wer(y_pred, y)
        self._num_examples += batch_size

    @sync_all_reduce("_wer", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('WordErrorRate must have at least one example before it can be computed.')
        return self._wer / self._num_examples

if __name__ == '__main__':
    pred = [
        'abc',
        'bc',
        'yes',
        'defg',
    ]
    tgt = [
        'bc',
        'bcd',
        'yes',
        'de',
    ]
    EOS = 'z'
    assert len(pred) == len(tgt)
    max_length_pred = max([len(text) for text in pred])
    max_length_tgt = max([len(text) for text in tgt])
    batch_size = len(pred)
    vocab_size = len(set(pred+tgt+[EOS]))
    pred = [list(text) + [EOS]*(max_length_pred + 1 - len(text)) for text in pred]
    tgt = [list(text) + [EOS]*(max_length_tgt + 1 - len(text)) for text in tgt]
    print('pred', pred)
    print('tgt', tgt)
    a = torch.LongTensor([list(map(ord, sample)) for sample in pred])
    b = torch.LongTensor([list(map(ord, sample)) for sample in tgt])
    print('a', a)
    print('b', b)

    cer = CharacterErrorRate(ord(EOS), batch_first=True)
    wer = WordErrorRate(ord(EOS), batch_first=True)

    cer.update((a,b))
    wer.update((a,b))

    _cer = cer.compute()
    _wer = wer.compute()
    print(_wer, _cer)
    assert _cer == ((1/2 + 1/3 + 0/3 + 2/2)/4)
    assert _wer == 3/4

    cer.update((a[:-1],b[:-1]))
    wer.update((a[:-1],b[:-1]))

    _cer = cer.compute()
    _wer = wer.compute()
    print(_wer, _cer)
    assert _cer == ((1/2 + 1/3 + 0/3 + 2/2 + 1/2 + 1/3 + 0/3)/7)
    assert _wer == 5/7
    print('OK!')
