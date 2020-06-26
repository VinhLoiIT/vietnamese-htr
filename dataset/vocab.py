import torch
import pandas as pd
from collections import Counter

from typing import List

class Vocab(object):

    @property
    def SOS(self):
        return '<start>'

    @property
    def EOS(self):
        return '<end>'

    @property
    def UNK(self):
        return '<unk>'

    @property
    def BLANK(self):
        return '<blank>'

    @property
    def SOS_IDX(self):
        return self.char2int(self.SOS)

    @property
    def EOS_IDX(self):
        return self.char2int(self.EOS)

    @property
    def UNK_IDX(self):
        return self.char2int(self.UNK)

    @property
    def BLANK_IDX(self):
        return self.char2int(self.BLANK)

    def __init__(self, add_blank: bool):
        labels = self.load_labels().apply(self.process_label)
        if not add_blank:
            labels = labels.apply(self.add_signals)
        counter = labels.apply(lambda word: Counter(word))
        counter = counter.sum()
        counter.update({self.UNK: 0})
        if add_blank:
            counter.update({self.BLANK: 0})
        self.alphabets = list(counter.keys())
        self.class_weight = torch.tensor([1. / counter[char] if counter[char] > 0 else 0 for char in self.alphabets])
        self.size = len(self.alphabets)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        raise NotImplementedError()

    def process_label(self, label: List[str]):
        '''
        Preprocess label (if needed), such as flattening out diacritical marks
        '''
        return label

    def process_label_invert(self, labels: List[List[str]]):
        '''
        Invert preprocessed label (if have), such as invert flattening diacritical marks
        '''
        return labels

    def add_signals(self, word: str):
        '''
        Add Start Of Sequence (SOS) and End Of Sequence (EOS) signals to string
        '''
        return sum([[self.SOS], list(word), [self.EOS]], [])

    def char2int(self, c: str) -> int:
        '''
        Convert character representation to index.
        Return index of UNK if unknow character
        '''
        try:
            return self.alphabets.index(c)
        except:
            return self.alphabets.index(self.UNK)

    def int2char(self, i: int) -> str:
        '''
        Convert an index to character representation
        '''
        return self.alphabets[i]
