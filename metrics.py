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
    def __init__(self, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

    def cer(self, pred, target):
        '''
        Parameters:
        -----------
        pred: [T_1,1]
        target: [T_2,1]

        Output:
        float cer
        '''
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        distance = ed.distance(pred.tolist(), target.tolist())
        return distance / len(target)

    @reinit__is_reduced
    def reset(self) -> None:
        self._cer = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y = output
        y_pred, pred_lengths = pad_packed_sequence(y_pred)
        y, lengths = pad_packed_sequence(y)
        y_pred = y_pred.topk(1,-1)[1]
        batch_cers = 0
        for i in range(len(lengths)):
            CER = self.cer(y_pred[:lengths[i], i], y[:lengths[i], i])
            batch_cers += CER
        batch_cers /= y_pred.shape[1]

        self._cer += batch_cers
        self._num_examples += len(lengths)

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
    def __init__(self, batch_first=False):
        super().__init__()
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
        y_pred, y = output
        y_pred, pred_lengths = pad_packed_sequence(y_pred)
        y, lengths = pad_packed_sequence(y)
        y_pred = y_pred.long()
        wers = 0
        for i in range(len(lengths)):
            wer = self.wer(y_pred[:lengths[i], i], y[:lengths[i], i])
            wers += wer
        wers /= y_pred.shape[1]

        self._wer += wers
        self._num_examples += len(lengths)

    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return self._wer / self._num_examples

if __name__ == '__main__':
    a = torch.LongTensor(list(map(ord,'Loi'))).unsqueeze(-1)
    b = torch.LongTensor(list(map(ord,'loi'))).unsqueeze(-1)
    cer = CharacterErrorRate.cer(None, a,b)
    wer = WordErrorRate.wer(None, a,b)
    print(wer, cer)
