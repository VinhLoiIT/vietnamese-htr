import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

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

    def char2int(self, c):
        raise NotImplementedError()

    def int2char(self, i):
        raise NotImplementedError()

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

        # images: [B, 3, H, W]
        max_image_row = max([image.size(1) for image in image_samples])
        max_image_col = max([image.size(2) for image in image_samples])
        self.images = torch.zeros(batch_size, 3, max_image_row, max_image_col)
        self.sizes = torch.zeros(batch_size, 2, dtype=torch.long)
        for i, image in enumerate(image_samples):
            image_row = image.shape[1]
            image_col = image.shape[2]
            self.sizes[i, 0], self.sizes[i, 1] = image_row, image_col
            self.images[i, :, :image_row, :image_col] = image

        self.lengths = torch.tensor([len(label) for label in label_samples])
        self.labels = pad_sequence(label_samples, batch_first=True)

    def pin_memory(self):
        self.images.pin_memory()
        self.labels.pin_memory()
        return self
