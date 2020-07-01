from typing import List, Tuple

import editdistance as ed
from pytorch_lightning.metrics import NumpyMetric

__all__ = [
    'compute_cer',
    'compute_wer',
    'CharacterErrorRate',
    'WordErrorRate',
]


def compute_cer(
    predicts: List[List[str]],
    targets: List[List[str]],
    indistinguish: bool
) -> Tuple[List[int], List[int]]:
    '''
    Calculate CER distance between two strings or two lists of strings

    Params:
    -------
    - predicts: List of predicted characters
    - targets: List of target characters
    - indistinguish: set to True to case-insensitive, or False to case-sensitive

    Returns:
    --------
    - distances: List of distances
    - n_references: List of the number of characters of targets
    '''
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = [ed.distance(predict, target) for predict, target in zip(predicts, targets)]
    num_references = list(map(len, targets))
    return distances, num_references


def compute_wer(
    predicts: List[List[str]],
    targets: List[List[str]],
    indistinguish: bool
) -> Tuple[List[int], List[int]]:
    '''
    Calculate CER distance between two strings or two lists of strings

    Params:
    -------
    - predicts: List of predicted characters
    - targets: List of target characters
    - indistinguish: set to True to case-insensitive, or False to case-sensitive

    Returns:
    --------
    - distances: List of distances
    - n_references: List of the number of words of targets
    '''
    assert type(predicts) == type(targets), 'predicts and targets must be the same type'
    assert len(predicts) == len(targets), 'predicts and targets must have the same length'

    distances = []
    num_references = []
    for predict, target in zip(predicts, targets):
        predict = ''.join(predict).split(' ')
        target = ''.join(target).split(' ')
        distances.append(ed.distance(predict, target))
        num_references.append(len(target))
    
    return distances, num_references


class CharacterErrorRate(NumpyMetric):
    def __init__(self, indistinguish: bool = False):
        super().__init__('CER')
        self.indistinguish = indistinguish

    def forward(self, predicts, targets):
        return compute_cer(predicts, targets, self.indistinguish)

class WordErrorRate(NumpyMetric):
    def __init__(self, indistinguish: bool = False):
        super().__init__('WER')
        self.indistinguish = indistinguish

    def forward(self, predicts, targets):
        return compute_wer(predicts, targets, self.indistinguish)
