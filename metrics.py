import torch
from torch.nn.utils.rnn import pad_packed_sequence

from ignite.engine import Engine, Events
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics import EpochMetric, Average
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import editdistance as ed

__all__ = [
    'CharacterErrorRate',
    'WordErrorRate',
]

class CharacterErrorRate(Metric):
    '''
    Calculates the CharacterErrorRate.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, vocab, batch_first=True, output_transform=lambda x: x, device=None):
        super().__init__(output_transform, device)
        self.batch_first = batch_first
        self.EOS_int = vocab.char2int(vocab.EOS)

    def calc_length(self, tensor):
        if not self.batch_first:
            tensor = tensor.transpose(0,1)

        lengths = []
        for sample in tensor.tolist():
            try:
                length = sample.index(self.EOS_int)
            except:
                length = len(sample)
            lengths.append(length)
        return lengths

    @reinit__is_reduced
    def reset(self) -> None:
        self._ed = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        '''
        output: ([B, T_1], [B, T2])
        '''
        y_pred, y = output

        if not self.batch_first:
            y_pred = y_pred.transpose(0,1) # [B,T_1]
            y = y.transpose(0,1) # [B,T_2]
        batch_size = y_pred.size(0)

        assert len(y_pred) == len(y)
        pred_lengths = self.calc_length(y_pred)
        target_lengths = self.calc_length(y)

        distances = torch.zeros(batch_size, dtype=torch.float)
        for i, (pred_length, tgt_length) in enumerate(zip(pred_lengths, target_lengths)):
            distance = ed.distance(y_pred[i, :pred_length].tolist(), y[i, :tgt_length].tolist())
            distance = float(distance) / target_lengths[i]
            distances[i] = distance

        self._ed += distances.sum()
        self._num_examples += batch_size

    @sync_all_reduce("_ed", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CharacterErrorRate must have at least one example before it can be computed.')
        return self._ed / self._num_examples

class WordErrorRate(Metric):
    '''
    Calculates the WordErrorRate.
    Notes:
    - When recognize at word-level, this metric is (1 - Accuracy)
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, vocab, batch_first=True, output_transform=lambda x: x):
        super().__init__(output_transform)
        self.EOS_int = vocab.char2int(vocab.EOS)
        self.batch_first = batch_first

    def calc_length(self, tensor):
        if not self.batch_first:
            tensor = tensor.transpose(0,1)

        lengths = []
        for sample in tensor.tolist():
            try:
                length = sample.index(self.EOS_int)
            except:
                length = len(sample)
            lengths.append(length)
        return lengths

    @reinit__is_reduced
    def reset(self) -> None:
        self._we = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        '''
        output: ([B, T_1], [B, T2])
        '''
        y_pred, y = output

        if not self.batch_first:
            y_pred = y_pred.transpose(0,1) # [B,T_1]
            y = y.transpose(0,1) # [B,T_2]
        batch_size = y_pred.size(0)

        assert len(y_pred) == len(y)
        pred_lengths = self.calc_length(y_pred)
        target_lengths = self.calc_length(y)

        distances = torch.zeros(batch_size, dtype=torch.float)
        for i, (pred_length, tgt_length) in enumerate(zip(pred_lengths, target_lengths)):
            distance = 0 if torch.equal(y_pred[i, :pred_length], y[i, :tgt_length]) else 1
            distances[i] = distance

        self._we += distances.sum()
        self._num_examples += batch_size

    @sync_all_reduce("_we", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('WordErrorRate must have at least one example before it can be computed.')
        return self._we / self._num_examples

    # def attach(self, engine: Engine, name: str) -> None:
    #     # restart every epoch
    #     if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
    #         engine.add_event_handler(Events.EPOCH_STARTED, self.started)
    #     # compute metric
    #     if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
    #         engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
    #     # apply running average
    #     engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class DummyVocab:
    EOS = 'z'
    def char2int(self, c):
        return ord(self.EOS)

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

    vocab = DummyVocab()
    cer = CharacterErrorRate(vocab, batch_first=True)
    wer = WordErrorRate(vocab, batch_first=True)

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
