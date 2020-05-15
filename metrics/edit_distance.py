import editdistance as ed
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = [
    'EditDistance',
    'CharacterErrorRate',
    'WordErrorRate',
]

class EditDistance(Metric):
    '''
    Calculates the EditDistance.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, logfile=None, output_transform=lambda x: x, device=None, is_global_ed=True, is_indistinguish_letter=True):
        super().__init__(output_transform, device)
        self._ed = 0.0
        self._num_references = 0 # Global ed: number characters in CER (or words in WER) of target
                                # Mean normalized ed: number examples
        self.is_global_ed = is_global_ed
        self.is_indistinguish_letter = is_indistinguish_letter
        self.log = logfile

        if self.log is not None:
            if isinstance(self.log, str):
                self.log = open(self.log, 'wt')

    def __del__(self):
        if self.log is not None:
            self.log.close()

    def compute_distance(self, predict: str, target: str):
        '''
        Compute edit distance between two strings and number reference (len(target))
        '''
        pass

    @reinit__is_reduced
    def reset(self) -> None:
        self._ed = 0.0
        self._num_references = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        '''
        output: (list of predict string, list of target string)
        '''
        y_pred, y = output
        assert len(y_pred) == len(y)
        
        batch_size = len(y)

        for i, (predict, target) in enumerate(zip(y_pred, y)):
            if self.is_indistinguish_letter:
                predict, target = predict.lower(), target.lower()
            distance, num_reference = self.compute_distance(predict, target)
            if self.log is not None:
                self.log.write(f'{"".join(predict)}|{"".join(target)}|{distance}\n')
            if self.is_global_ed:
                self._ed += distance
                self._num_references += num_reference
            else:
                self._ed += distance / num_reference
                self._num_references += 1

    @sync_all_reduce("_ed", "_num_references")
    def compute(self):
        if self._num_references == 0:
            raise NotComputableError('EditDistance must have at least one example before it can be computed.')
        return self._ed / self._num_references

class CharacterErrorRate(EditDistance):
    '''
    Calculates the CharacterErrorRate.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, logfile=None, output_transform=lambda x: x, device=None, is_global_ed=True, is_indistinguish_letter=False):
        super().__init__(logfile, output_transform, device, is_global_ed, is_indistinguish_letter)

    def compute_distance(self, predict: str, target: str):
        '''
        Compute edit distance between two strings
        '''
        distance = ed.distance(predict, target)
        return distance, len(target)

class WordErrorRate(EditDistance):
    '''
    Calculates the WordErrorRate.
    Notes:
    - When recognize at word-level, this metric is (1 - Accuracy)
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    '''
    def __init__(self, logfile=None, output_transform=lambda x: x, device=None, is_global_ed=True, is_indistinguish_letter=True):
        super().__init__(logfile, output_transform, device, is_global_ed, is_indistinguish_letter)

    def compute_distance(self, predict: str, target: str):
        '''
        Compute edit distance between two strings
        '''
        predict = ''.join(predict).split(' ')
        target = ''.join(target).split(' ')
        distance = ed.distance(predict, target)
        return distance, len(target)
