import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
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
        labels = self.load_labels().apply(self.process_label).apply(self.add_signals)
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

class CollateWrapper:
    def __init__(self, batch):
        '''
        Shapes:
        -------
        batch: list of 2-tuples:
            - image: tensor of [C, H, W]
            - label: tensor of [T*] where T* varies from labels and includes '<start>' and '<end>' at both ends
        Returns:
        --------
        - images: tensor of [B, C, max_H, max_W]
        - size: tensor of [B, 2]
        - labels: tensor of [B,max_length,1]
        - length: tensor of [B]
        '''

        batch_size = len(batch)
        batch.sort(key=lambda sample: len(sample[1]), reverse=True)
        image_samples, label_samples = list(zip(*batch))

        self.images = self.collate_images(image_samples)
        self.lengths = torch.tensor([len(label) for label in label_samples])
        self.labels = pad_sequence(label_samples, batch_first=True)

    def collate_images(self, image_samples: List[torch.Tensor]) -> torch.Tensor:
        # images: [B, 3, H, W]
        max_image_row = max([image.size(1) for image in image_samples])
        max_image_col = max([image.size(2) for image in image_samples])
        images = torch.zeros(len(image_samples), 3, max_image_row, max_image_col)
        for i, image in enumerate(image_samples):
            image_row = image.shape[1]
            image_col = image.shape[2]
            images[i, :, :image_row, :image_col] = image
        return images

    def pin_memory(self):
        self.images.pin_memory()
        self.labels.pin_memory()
        return self
