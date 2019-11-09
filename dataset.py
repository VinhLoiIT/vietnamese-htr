import pandas as pd
import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VNOnDB(Dataset):
    def __init__(self, root_dir, dataframe, image_transform=None, label_transform=None):
        self.root_dir = root_dir
        
        self.df = dataframe
        self.image_transform = image_transform
        self.label_transform = label_transform
        
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
    
def to_batch(samples):
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