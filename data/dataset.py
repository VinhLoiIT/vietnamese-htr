import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

SOS_CHAR = '<start>' # start of sequence character
EOS_CHAR = '<end>' # end of sequence character
PAD_CHAR = '<pad>' # padding character

class Vocab(object):
    def __init__(self, dataset):
        alphabets = get_alphabets(dataset)
        self.char2int = dict((c, i) for i, c in enumerate(alphabets))
        self.int2char = dict((i, c) for i, c in enumerate(alphabets))
        self.vocab_size = len(alphabets)
    
    def __call__(self, samples):
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
            label_int = torch.tensor([self.char2int[char] for char in label]).view(-1, 1) # [T, 1]
            labels[:, i] = label_int
            
        labels_onehot = torch.zeros(max(label_lengths), batch_size, self.vocab_size, dtype=torch.long) # [max_T, B, vocab_size]
        for label_i, label in enumerate(label_samples):
            for char_i, char in enumerate(label):
                char_int = self.char2int[char]
                onehot = torch.zeros(self.vocab_size, dtype=torch.long)
                onehot[char_int] = 1
                labels_onehot[char_i, label_i] = onehot

        return images, labels, labels_onehot, torch.tensor(label_lengths).view(-1, 1)

def get_alphabets(dataset):
    if dataset == 'vnondb':
        alphabets = pd.read_csv('./data/VNOnDB/all_word.csv', sep='\t', keep_default_na=False, index_col=0)
        alphabets = sorted(list(set.union(*alphabets.label.apply(set))) + [SOS_CHAR, EOS_CHAR, PAD_CHAR])
        return alphabets
    elif dataset == 'rimes':
        with open ('./data/RIMES/groundtruth_training_icdar2011.txt') as f:
            content = f.readlines()
        alphabets = sorted(list(set(''.join([x.strip().split(' ')[-1] for x in content])))+[SOS_CHAR, EOS_CHAR, PAD_CHAR])
        return alphabets
    elif dataset == 'iam':
        raise NotImplementedError()
    else:
        raise ValueError('should be {}'.format(['vnondb', 'rimes', 'iam']))

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

class RIMES(Dataset):
    def __init__(self, image_folder, groundtruth_txt, image_transform=None):
        with open (groundtruth_txt, encoding='utf-8-sig') as f:
            content = f.readlines()
        self.content = [x.strip() for x in content]
        self.image_folder = image_folder
        self.image_transform = image_transform

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.content[idx].split(' ')[0])
        image = Image.open(image_path)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = self.content[idx].split(' ')[1]
        label = [SOS_CHAR] + list(label) + [EOS_CHAR]
            
        return image, label