import torch
import pandas as pd
import os
from PIL import Image
from .vocab import Vocab

class HTRDataset(torch.utils.data.Dataset):

    def __init__(self,
        vocab: Vocab,
        image_folder: str,
        csv: str,
        image_transform,
    ):
        self.vocab = vocab
        self.image_transform = image_transform
        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False)
        self.df.iloc[:, 0] = self.df.iloc[:, 0].apply(lambda filename: os.path.join(image_folder, filename))
        self.df.iloc[:, 1] = self.df.iloc[:, 1].apply(self.vocab.process_label).apply(self.vocab.add_signals)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = Image.open(image_path).convert('L')
        image = self.image_transform(image)
        
        label = torch.tensor(list(map(self.vocab.char2int, self.df.iloc[idx, 1])))
            
        return image, label
