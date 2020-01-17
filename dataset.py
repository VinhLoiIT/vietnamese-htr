import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

SOS_CHAR = '<start>' # start of sequence character
EOS_CHAR = '<end>' # end of sequence character
PAD_CHAR = '<pad>' # padding character

alphabets = pd.read_csv('./data/VNOnDB/all_word.csv', sep='\t', keep_default_na=False, index_col=0)
alphabets = sorted(list(set.union(*alphabets.label.apply(set))) + [SOS_CHAR, EOS_CHAR, PAD_CHAR])

char2int = dict((c, i) for i, c in enumerate(alphabets))
int2char = dict((i, c) for i, c in enumerate(alphabets))
vocab_size = len(alphabets)


class VNOnDB(Dataset):
    def __init__(self, image_folder, csv, image_transform=None):
        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False, index_col=0)
        self.image_folder = image_folder
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.df['id'][idx]+'.png')
        image = Image.open(image_path)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = self.df['label'][idx]
        label = [SOS_CHAR] + list(label) + [EOS_CHAR]
            
        return image, label
    
def get_dataset(dataset, transform):
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('Should be: ' + str(['train', 'test', 'val']))

    if dataset == 'test':
        return VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv', transform)
    if dataset == 'train':
        return VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', transform)
    if dataset == 'val':
        return VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', transform)
    
def get_data_loader(dataset, batch_size, transform=None, debug=False):
    data = get_dataset(dataset, transform)
    if debug:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=12,
                            sampler=SubsetRandomSampler(np.random.permutation(min(batch_size * 5, len(data)))))
    else:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=12)
    return loader
    
def collate_fn(samples):
    '''
    :param samples: list of tuples:
        - image: tensor of [C, H, W]
        - label: list of characters including '<start>' and '<end>' at both ends
    :returns:
        - images: tensor of [B, C, H, W]
        - labels: tensor of [max_T, B, 1]
        - lengths: tensor of [B, 1]
    '''
    batch_size = len(samples)
    samples.sort(key=lambda sample: len(sample[1]), reverse=True)
    image_samples, label_samples = list(zip(*samples))

    # images: [B, 3, H, W]
    max_image_row = max([image.size(1) for image in image_samples])
    max_image_col = max([image.size(2) for image in image_samples])
    images = torch.ones(batch_size, 3, max_image_row, max_image_col)
    for i, image in enumerate(image_samples):
        image_row = image.shape[1]
        image_col = image.shape[2]
        images[i, :, :image_row, :image_col] = image

    label_lengths = [len(label) for label in label_samples]
    max_length = max(label_lengths)
    label_samples = [label + [PAD_CHAR] * (max_length - len(label)) for label in label_samples]
    
    labels = torch.zeros(max(label_lengths), batch_size, 1, dtype=torch.long) # [max_T, B, 1]
    for i, label in enumerate(label_samples):
        label_int = torch.tensor([char2int[char] for char in label]).view(-1, 1) # [T, 1]
        labels[:, i] = label_int
        
    labels_onehot = torch.zeros(max(label_lengths), batch_size, vocab_size, dtype=torch.long) # [max_T, B, vocab_size]
    for label_i, label in enumerate(label_samples):
        for char_i, char in enumerate(label):
            char_int = char2int[char]
            onehot = torch.zeros(vocab_size, dtype=torch.long)
            onehot[char_int] = 1
            labels_onehot[char_i, label_i] = onehot

    return images, labels, labels_onehot, torch.tensor(label_lengths).view(-1, 1)
