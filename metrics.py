import torch
from torch.nn.utils.rnn import pad_packed_sequence

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
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
    def __init__(self, EOS_int, output_transform=None, batch_first=False):
        super().__init__()
        self.EOS_int = EOS_int
        self.output_transform = output_transform
        self.batch_first = batch_first

    def cer(self, pred, target):
        '''
        Parameters:
        -----------
        pred: [T_1]
        target: [T_2]

        Output:
        float cer
        '''
        distance = ed.distance(pred.tolist(), target.tolist())
        return distance / len(target)

    @reinit__is_reduced
    def reset(self) -> None:
        self._cer = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        if self.output_transform is not None:
            output = self.output_transform(output)
        y_pred, y = output

        if not self.batch_first:
            y_pred = y_pred.transpose(0,1) # [B,T_1]
            y = y.transpose(0,1) # [B,T_2]
        y_pred = y_pred.squeeze(-1)
        y = y.squeeze(-1)
        batch_size = y_pred.size(0)

        y_pred_lengths = []
        for sample in y_pred.tolist():
            try:
                end = sample.index(self.EOS_int)
            except:
                end = None
            y_pred_lengths.append(end)

        y_lengths = []
        for sample in y.tolist():
            try:
                end = sample.index(self.EOS_int)
            except:
                end = None
            y_lengths.append(end)

        batch_cers = 0
        for i in range(batch_size):
            CER = self.cer(y_pred[i, :y_pred_lengths[i]], y[i, :y_lengths[i]])
            batch_cers += CER

        self._cer += batch_cers
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
    def __init__(self, EOS_int, output_transform=None, batch_first=False):
        super().__init__()
        self.EOS_int = EOS_int
        self.output_transform = output_transform
        self.batch_first = batch_first

    def wer(self, pred, target):
        '''
        Parameters:
        -----------
        pred: [T_1,1]
        target: [T_2,1]

        Output:
        float wer
        '''
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        if torch.equal(pred, target):
            return 0
        else:
            return 1

    @reinit__is_reduced
    def reset(self) -> None:
        self._wer = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        if self.output_transform is not None:
            output = self.output_transform(output)
        y_pred, y = output


        if not self.batch_first:
            y_pred = y_pred.transpose(0,1) # [B,T_1]
            y = y.transpose(0,1) # [B,T_2]
        y_pred = y_pred.squeeze(-1)
        y = y.squeeze(-1)
        batch_size = y_pred.size(0)

        y_pred_lengths = []
        for sample in y_pred.tolist():
            try:
                end = sample.index(self.EOS_int)
            except:
                end = None
            y_pred_lengths.append(end)

        y_lengths = []
        for sample in y.tolist():
            try:
                end = sample.index(self.EOS_int)
            except:
                end = None
            y_lengths.append(end)

        wers = 0
        for i in range(batch_size):
            WER = self.wer(y_pred[i, :y_pred_lengths[i]], y[i, :y_lengths[i]])
            wers += WER

        self._wer += wers
        self._num_examples += batch_size

    @sync_all_reduce("_wer", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return self._wer / self._num_examples

if __name__ == '__main__':
    a = torch.LongTensor(list(map(ord,'Loi')))
    b = torch.LongTensor(list(map(ord,'loi')))
    cer = CharacterErrorRate.cer(None, a,b)
    wer = WordErrorRate.wer(None, a,b)
    print(wer, cer)
