from typing import Callable, Optional, Sequence, Union

import editdistance as ed
import torch
from ignite.engine import Engine, Events
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from torch.nn.utils.rnn import pad_packed_sequence

__all__ = [
    'CharacterErrorRate',
    'WordErrorRate',
    'Running',
]

class Running(Metric):
    
    _required_output_keys = None

    def __init__(
        self,
        src: Optional[Metric] = None,
        output_transform: Optional[Callable] = None,
        epoch_bound: bool = True,
        reset_interval: Optional[Union[int, None]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            if device is not None:
                raise ValueError("Argument device should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError(
                    "Argument output_transform should not be None if src corresponds "
                    "to the output of process function."
                )
            self._get_src_value = self._get_output_value
            self.update = self._output_update

        self.epoch_bound = epoch_bound
        self.reset_interval = reset_interval
        super(Running, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.src.reset()

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        # Implement abstract method
        pass

    def compute(self) -> Union[torch.Tensor, float]:
        return self._get_metric_value()

    def attach(self, engine: Engine, name: str):
        if self.epoch_bound:
            # restart average every epoch
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if self.reset_interval:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.reset_interval), self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self) -> Union[torch.Tensor, float]:
        return self.src.compute()

    @sync_all_reduce("src")
    def _get_output_value(self) -> Metric:
        return self.src

    def _metric_iteration_completed(self, engine: Engine) -> None:
        self.src.iteration_completed(engine)

    @reinit__is_reduced
    def _output_update(self, output: Metric) -> None:
        self.src = output

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
