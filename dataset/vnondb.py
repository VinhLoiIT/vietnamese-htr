import os
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CollateWrapper, Vocab


class VNOnDBVocab(Vocab):
    def __init__(self):
        super().__init__()
        df = pd.read_csv('./data/VNOnDB/train_word.csv', sep='\t', keep_default_na=False, index_col=0)
        df['counter'] = df['label'].apply(lambda word: Counter([self.SOS] + list(word) + [self.EOS]))
        counter = df['counter'].sum()
        counter.update({self.UNK: 0})
        self.alphabets = list(counter.keys())
        self.class_weight = torch.tensor([1. / counter[char] if counter[char] > 0 else 0 for char in self.alphabets])
        self.size = len(self.alphabets)

    def char2int(self, c):
        try:
            return self.alphabets.index(c)
        except:
            return self.alphabets.index(self.UNK)

    def int2char(self, i):
        return self.alphabets[i]

class VNOnDB(Dataset):
    vocab = None

    def __init__(self, image_folder, csv, image_transform=None):
        if self.vocab is None:
            self.vocab = VNOnDBVocab()
        self.image_transform = image_transform

        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False, index_col=0)
        self.df['id'] = self.df['id'].apply(lambda id: os.path.join(image_folder, id+'.png'))
        self.df['label'] = self.df['label'].apply(lambda x: [self.vocab.SOS] + list(x) + [self.vocab.EOS])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        image = Image.open(image_path).convert('L')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = torch.tensor(list(map(self.vocab.char2int, self.df['label'][idx])))
            
        return image, label

if __name__ == '__main__':

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    dataset = VNOnDB('./data/VNOnDB/train_word', './data/VNOnDB/train_word.csv', tf)
    print(len(dataset))
    print(dataset.vocab.size)
    print(dataset.vocab.alphabets)
    print(dataset.vocab.class_weight)

    loader = DataLoader(dataset, min(8, len(dataset)), False,
                        collate_fn=lambda batch: CollateWrapper(batch))
    batch = next(iter(loader))
    print(batch.images)
    print(batch.sizes)
    print(batch.labels)
    print(batch.lengths)
