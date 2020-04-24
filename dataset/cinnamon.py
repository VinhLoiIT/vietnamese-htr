from .vocab import Vocab
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from PIL import Image
import os

class CinnamonVocab(Vocab):
    def __init__(self, train_csv: str, add_blank):
        self._train_csv = train_csv
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        df = pd.read_csv(self._train_csv, sep='\t', keep_default_na=False)
        labels = df['label']
        return labels

class Cinnamon(Dataset):
    def __init__(self, vocab, image_folder, csv, image_transform=None):
        self.vocab = vocab
        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False)
        self.df['image'] = self.df['image'].apply(lambda path: os.path.join(image_folder, path))
        self.df['label'] = self.df['label'].apply(self.vocab.process_label).apply(self.vocab.add_signals)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.df['image'].iloc[index])
        if self.image_transform:
            image = self.image_transform(image)

        text = self.df['label'].iloc[index]
        text = torch.tensor(list(map(self.vocab.char2int, text)))

        return image, text
