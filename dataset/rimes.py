import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset

import pandas as pd
from .vocab import CollateWrapper, Vocab


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


class RIMESLine(Dataset):
    def __init__(self, vocab, root_folder, csv, image_transform=None):
        self.vocab = vocab
        self.image_transform = image_transform

        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False)
        self.df['filename'] = self.df['filename'].apply(lambda path: os.path.join(root_folder, path))
        self.df['label'] = self.df['label'].apply(self.vocab.process_label).apply(self.vocab.add_signals)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label, bottom, top, right, left = self.df.iloc[idx]
        image = Image.open(image_path).convert('L')
        image = image.crop((left, top, right, bottom))

        if self.image_transform:
            image = self.image_transform(image)

        label = torch.tensor(list(map(self.vocab.char2int, label)))

        return image, label


if __name__ == '__main__':

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    # dataset = RIMES('./data/RIMES/data_test', './data/RIMES/grount_truth_test_icdar2011.txt', transform)
    dataset = RIMES('./data/RIMES/trainingsnippets_icdar/training_WR', './data/RIMES/groundtruth_training_icdar2011.txt', transform)
    # dataset = RIMES('./data/RIMES/validationsnippets_icdar/testdataset_ICDAR', './data/RIMES/ground_truth_validation_icdar2011.txt', transform)
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
