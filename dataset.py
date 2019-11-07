import pandas as pd
import os
from PIL import Image
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