import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
        images = torch.zeros(batch_size, 3, max_image_row, max_image_col)
        for i, image in enumerate(image_samples):
            image_row = image.shape[1]
            image_col = image.shape[2]
            images[i, :, :image_row, :image_col] = image

        label_lengths = torch.tensor([len(label) for label in label_samples])
        max_length = label_lengths.max().item()

        label_tensors = []
        for label in label_samples:
            label = [self.char2int[char] for char in label]
            label = torch.tensor(label)
            label = F.pad(label, (0, max_length - len(label)), 'constant', self.char2int[PAD_CHAR])
            label_tensors.append(label)
        label_tensors = torch.stack(label_tensors, dim=0)

        return images, label_tensors, label_lengths

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
        alphabets = set()
        with open ('./data/IAM/words.txt') as f:
            for line in f:
                if not line or line.startswith('#'): # Comment in txt file
                    continue
                line_split = line.strip().split(' ')
                gt_text = ' '.join(line_split[8:]) # Word, but contains spaces
                alphabets = alphabets.union(set(list(gt_text)))          
        alphabets = sorted(list(alphabets)+[SOS_CHAR, EOS_CHAR, PAD_CHAR])
        return alphabets
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
    
class IAM(Dataset):
    def __init__(self, partition_split, image_transform=None):
        self.image_folder = './data/IAM/words'
        self.annotation_file = './data/IAM/words.txt'
        with open (partition_split) as f:
            partition_folder = f.readlines()
        self.partition_folder = [x.strip() for x in partition_folder]
        self.image_transform = image_transform

        self.content = [] # [[path, label]*N]
        with open (self.annotation_file) as f:
            for line in f:
                if not line or line.startswith('#'): # comment in txt file
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err': # er: segmentation of word can be bad
                    continue

                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join(label_dir, sub_label_dir, fn)

                gt_text = ' '.join(line_split[8:]) # Word, but contains spaces

                if sub_label_dir in self.partition_folder:
                    self.content.append([img_path, gt_text])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.content[idx][0])
        image = Image.open(image_path)

        if self.image_transform:
            image = self.image_transform(image)

        label = self.content[idx][1]
        label = [SOS_CHAR] + list(label) + [EOS_CHAR]

        return image, label