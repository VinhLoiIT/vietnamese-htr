import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CollateWrapper, Vocab


class RIMESVocab(Vocab):
    def __init__(self):
        super().__init__()

        counter = Counter()
        with open ('./data/RIMES/groundtruth_training_icdar2011.txt') as f:
            content = f.readlines()
        [counter.update([self.SOS] + list(x.strip().split(' ')[-1]) + [self.EOS]) for x in content]
        self.alphabets = list(counter.keys())
        self.class_weight = [1. / counter[char] if counter[char] > 0 else 0 for char in self.alphabets]
        self.size = len(self.alphabets)

    def char2int(self, c):
        try:
            return self.alphabets.index(c)
        except:
            return self.alphabets.index(self.UNK)

    def int2char(self, i):
        return self.alphabets[i]

class RIMES(Dataset):
    vocab = None
    def __init__(self, image_folder, groundtruth_txt, image_transform=None):
        if self.vocab is None:
            self.vocab = RIMESVocab()
        with open (groundtruth_txt, encoding='utf-8-sig') as f:
            content = f.readlines()
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
