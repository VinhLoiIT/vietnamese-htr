import numpy as np
import pandas as pd
import os
from PIL import Image
import pdb
import torch

from torch.utils.data import Dataset


class VNOnDBData:

    sos_char = '<start>' # start of sequence character
    eos_char = '<end>' # end of sequence character
    pad_char = '<pad>' # padding character

    def __init__(self, all_csv):
        df = pd.read_csv(all_csv, sep='\t', keep_default_na=False, index_col=0)
        self.alphabets = list(set.union(*df.label.apply(set)))
        self.alphabets += [VNOnDBData.sos_char, VNOnDBData.eos_char, VNOnDBData.pad_char]

        self.char2int = dict((c, i) for i, c in enumerate(self.alphabets))
        self.int2char = dict((i, c) for i, c in enumerate(self.alphabets))
        self.vocab_size = len(self.alphabets)
    
    def encode(self, characters: list):
        """
        Encode a list of characters to indexes using alphabets
        :returns: np.array of shape [len(characters), 1]
        """
        return np.array([[self.char2int[char]] for char in characters])

    @staticmethod
    def onehot2int(one_hot_vectors):
        return np.array([[np.argmax(vector)] for vector in one_hot_vectors])

    def int2onehot(self, character_ints):
        result = torch.zeros((len(character_ints), len(self.alphabets)))
        for i, char_int in enumerate(character_ints):
            result[i, char_int] = 1
        return result


    def decode(self, one_hot_vectors):
        string = ''.join(self.int2char[np.argmax(vector)] for vector in one_hot_vectors)
        string = string.replace(VNOnDBData.eos_char, '')
        return string

    # def alphabets(csv_file):
    #     if csv_file is None:
    #         lower_vowels_with_dm = u'áàảãạắằẳãặâấầẩẫậíìỉĩịúùủũụưứừửữựéèẻẽẹêếềểễệóòỏõọơớờởỡợôốồổỗộyýỳỷỹỵ'
    #         upper_vowels_with_dm = lower_vowels_with_dm.upper()
    #         lower_without_dm = u'abcdefghijklmnopqrstuvwxyzđ'
    #         upper_without_dm = lower_without_dm.upper()
    #         digits = '1234567890'
    #         symbols = '?/*+-!,."\':;#%&()[]'
    #         alphabets = lower_vowels_with_dm + lower_without_dm + upper_vowels_with_dm + upper_without_dm + digits + symbols
    #         alphabets = list(alphabets)
    

class VNOnDB(Dataset):
    def __init__(self, root_dir, csv_file, all_data: VNOnDBData, image_transform=None):
        self.root_dir = root_dir
        
        self.df = pd.read_csv(csv_file, sep='\t', keep_default_na=False, index_col=0)
        self.image_transform = image_transform
        self.all_data = all_data

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root_dir, self.df['id'][idx]+'.png')
        image = Image.open(image_path)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = self.df['label'][idx]
        label = list(label) + [VNOnDBData.eos_char]
        label = torch.tensor([self.all_data.char2int[char] for char in label], dtype=torch.long)
        onehot = self.all_data.int2onehot(label)
            
        return image, label, onehot

def collate_fn(samples):
    '''
    :param samples: list of tuples:
        - image: tensor of [C, H, W]
        - label: tensor of [T]
        - onehot: tensor of [T, V]
    :returns:
        - images: tensor of [B, C, H, W]
        - labels: tensor of [B, max_T]
        - labels_onehot: tensor of [B, max_T, V]
        - lengths: tensor of [B, 1]
    '''
    batch_size = len(samples)
    samples.sort(key=lambda sample: len(sample[1]), reverse=True)
    image_samples, label_samples, onehot_samples = list(zip(*samples))

    # images: [B, 3, H, W]
    # image: [3, H, W] - grayscale
    max_image_row = max([image.size(1) for image in image_samples])
    max_image_col = max([image.size(2) for image in image_samples])
    images = torch.ones(batch_size, 3, max_image_row, max_image_col)
    for i, image in enumerate(image_samples):
        image_row = image.shape[1]
        image_col = image.shape[2]
        images[i, :, :image_row, :image_col] = image

    label_lengths = torch.tensor([[len(label)] for label in label_samples]) # [B, 1]

    labels = torch.zeros(batch_size, max(label_lengths).item(), dtype=torch.long) # [B, max_T]
    for i, label in enumerate(label_samples):
        labels[i, :label_lengths[i].item()] = label

    vocab_size = onehot_samples[0].size(1)
    labels_onehot = torch.zeros(batch_size, max(label_lengths), vocab_size, dtype=torch.long) # [B, max_T, V]
    for i, onehot in enumerate(onehot_samples):
        labels_onehot[i, :label_lengths[i].item()] = onehot

    labels = labels.transpose(0, 1) # [max_T, B]
    labels_onehot = labels_onehot.transpose(0, 1) # [max_T, B, V]

    return images, labels, labels_onehot, label_lengths
