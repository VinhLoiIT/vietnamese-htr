import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset

import pandas as pd
from .vocab import Vocab


class RIMESVocab(Vocab):
    def __init__(self, add_blank: bool):
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        with open ('./data/RIMES/groundtruth_training_icdar2011.txt') as f:
            content = f.readlines()
        content = [line.strip().split(' ')[-1] for line in content]
        label = pd.Series(content, dtype=str)
        return label

class RIMES(Dataset):
    def __init__(self, vocab, image_folder, groundtruth_txt, image_transform=None):
        self.vocab = vocab
        with open (groundtruth_txt, encoding='utf-8-sig') as f:
            content = f.readlines()

        for i, line in enumerate(content):
            if len(line.strip().split(' ')) <= 1:
                print(f'Remove line {i}: {line}')
        content = [line for line in content[1:] if len(line.strip().split(' ')) > 1]
        self.image_paths, self.labels = list(zip(*[x.strip().split(' ') for x in content]))
        self.image_paths = [os.path.join(image_folder, path) for path in self.image_paths]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)

        label = [self.vocab.SOS] + list(self.labels[idx]) + [self.vocab.EOS]
        label = torch.tensor(list(map(self.vocab.char2int, label)))

        return image, label


class RIMESLineVocab(Vocab):
    def __init__(self, add_blank: bool):
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        train_df = pd.read_csv('data/RIMES_Line/train.csv', sep='\t')
        return train_df['label'].astype(str)
