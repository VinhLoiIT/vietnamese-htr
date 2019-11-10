import numpy as np
import pandas as pd
import os
from PIL import Image
import pdb
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VNOnDB(Dataset):
    def __init__(self, root_dir, csv_file, image_transform=None, label_transform=None):
        self.root_dir = root_dir
        
        self.df = pd.read_csv(csv_file, sep='\t')
        self.image_transform = image_transform
        self.label_transform = label_transform

        self.int2char = VNOnDB.alphabets()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root_dir, self.df['id'][idx]+'.png')
        image = Image.open(image_path)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = self.df['label'][idx]
        if self.label_transform:
            label = self.label_transform(label)
            
        return image, label

class VNOnDBData:
    def __init__(self, all_csv):
        self.alphabets = VNOnDBData.get_alphabets(all_csv)
        self.char2int = dict((c, i) for i, c in enumerate(self.alphabets))
        self.int2char = dict((i, c) for i, c in enumerate(self.alphabets))

    @staticmethod
    def get_alphabets(csv_file):
        alphabets = set()
        df = pd.read_csv(csv_file, sep='\t')
        words_list = df.loc[:, 'label'].astype(str)
        for word in words_list.values:
            alphabets = letters.union(set(list(word)))
        alphabets += ['<start>', '<end>']
        return alphabets
    
    def encode(self, characters: list):
        """
        Encode a string to one hot vector using alphabets
        """
        return np.array([[self.char2int[char]] for char in characters])

    def to_one_hot(self, encoded_characters):
        result = np.zeros((len(encoded_characters), len(alphabets)))
        for i, char_int in enumerate(encoded_characters):
            result[i, char_int] = 1
        return result


    def decode(self, one_hot_vectors):
        string = ''.join(int_to_char[np.argmax(vector)] for vector in one_hot_vectors)
        string = string.replace(eos_char, '')
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

    

    
def to_batch(samples):
    pdb.set_trace()
    batch_size = len(samples)
    image_samples, label_samples = list(zip(*samples))
    # image_samples: list of [C, H, W]
    # label_samples: list of [1, T, V]

    # batch_image: [B, 3, H, W]
    # image: [3, H, W] - grayscale
    max_image_row = max([image.size(1) for image in image_samples])
    max_image_col = max([image.size(2) for image in image_samples])
    batch_image = torch.ones(batch_size, 3, max_image_row, max_image_col)
    for i, image in enumerate(image_samples):
        image_row = image.shape[1]
        image_col = image.shape[2]
        batch_image[i, :, :image_row, :image_col] = image

    # batch_label: [T, B, 1]
    label_lengths = np.array([len(label) for label in label_samples])
    max_length = label_lengths.max()
    
    label_lengths = torch.from_numpy(label_lengths).unsqueeze(-1) # [B, 1]

    batch_label = np.zeros((batch_size, max_length, 1)) # [B, T, 1]
    for i, label in enumerate(label_samples): # label: list
        batch_label[i, :len(label)] = encode(label)
#     pdb.set_trace()
    batch_label = torch.from_numpy(batch_label).long() # [B, T, 1]
    batch_label_one_hot = torch.stack([torch.from_numpy(to_one_hot(label.numpy())) for label in batch_label]) # [B, T, V]

    # sort by decreasing lengths
    label_lengths, sorted_idx = label_lengths.squeeze(-1).sort(descending=True) # [B, 1]
    batch_image = batch_image[sorted_idx] # [B, C, H, W]
    batch_label = batch_label[sorted_idx].transpose(0, 1) # [T, B, 1]
    batch_label_one_hot = batch_label_one_hot[sorted_idx].transpose(0, 1) # [T, B, V]

    return batch_image, batch_label, batch_label_one_hot, label_lengths

def get_transforms():
    image_transform = transforms.Compose([
        transforms.Resize((320, 480)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    label_transform = transforms.Compose([
        transforms.Lambda(lambda label: list(label) + [eos_char]),
    ])
    
    return image_transform, label_transform