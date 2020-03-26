import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = [
    'WordErrorRate',
]

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
