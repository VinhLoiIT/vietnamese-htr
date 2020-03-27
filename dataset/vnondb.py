import os
from collections import Counter
import re
import unicodedata

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CollateWrapper, Vocab


class VNOnDBVocab(Vocab):
    def __init__(self):
        super().__init__()
        flattening = Flattening()
        df = pd.read_csv('./data/VNOnDB/train_word.csv', sep='\t', keep_default_na=False, index_col=0)
        df['counter'] = df['label'].apply(lambda word: Counter([self.SOS] + re.findall(r'[\w]|<.*?>', flattening.flatten_word(word)) + [self.EOS]))
        counter = df['counter'].sum()
        counter.update({self.UNK: 0})
        self.alphabets = list(counter.keys())
        self.class_weight = torch.tensor([1. / counter[char] if counter[char] > 0 else 0 for char in self.alphabets])
        self.size = len(self.alphabets)

    def char2int(self, c):
        try:
            return self.alphabets.index(c)
        except:
            return self.alphabets.index(self.UNK)

    def int2char(self, i):
        return self.alphabets[i]

class VNOnDB(Dataset):
    vocab = None

    def __init__(self, image_folder, csv, image_transform=None):
        if self.vocab is None:
            self.vocab = VNOnDBVocab()
        self.image_transform = image_transform
        flattening = Flattening()

        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False, index_col=0)
        self.df['id'] = self.df['id'].apply(lambda id: os.path.join(image_folder, id+'.png'))
        self.df['label'] = self.df['label'].apply(lambda x: [self.vocab.SOS] + re.findall(r'[\w]|<.*?>', flattening.flatten_word(x)) + [self.vocab.EOS])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        image = Image.open(image_path).convert('L')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = torch.tensor(list(map(self.vocab.char2int, self.df['label'][idx])))
            
        return image, label

class Flattening:
    def __init__(self, accent_letters=None):
        self.circumflex_unicodes = ['00C2', '1EA0', '00E2', '1EA1', '00CA', '00EA', '00D4', '00F4'] # â, Â, Ê, ...
        self.breve_unicodes = ['0102', '0103'] # ă, Ă
        self.underdot_unicodes = ['1EA0', '1EA1', '1EB8', '1EB9', '1ECC', '1ECD']
        if accent_letters==None:
            self.accent_letters = 'À Á Ả Ã Ạ Â Ầ Ấ Ẩ Ẫ Ậ Ă Ằ Ắ Ẳ Ẵ Ặ à á ả ã ạ â ầ ấ ẩ ẫ ậ ă ằ ắ ẳ ẵ ặ\
            È É Ẻ Ẽ Ẹ Ê Ề Ế Ể Ễ Ệ è é ẻ ẽ ẹ ê ề ế ể ễ ệ\
            Ì Í Ỉ Ĩ Ị ì í ỉ ĩ ị\
            Ò Ó Ỏ Õ Ọ Ô Ồ Ố Ổ Ỗ Ộ Ơ Ờ Ớ Ở Ỡ Ợ ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ\
            Ù Ú Ủ Ũ Ụ Ư Ừ Ứ Ử Ữ Ự ù ú ủ ũ ụ ư ừ ứ ử ữ ự\
            Ỳ Ý Ỷ Ỹ Ỵ ỳ ý ỷ ỹ ỵ'
            self.accent_letters = self.accent_letters.split()
    
    def get_unaccent(self, letter):
        letter = re.sub(u'[àáảãạâầấẩẫậăằắẳẵặ]', 'a', letter)
        letter = re.sub(u'[ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶ]', 'A', letter)
        letter = re.sub(u'[èéẹẻẽêềếệểễ]', 'e', letter)
        letter = re.sub(u'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', letter)
        letter = re.sub(u'[òóọỏõôồốộổỗ]', 'o', letter)
        letter = re.sub(u'[ÒÓỌỎÕÔỒỐỘỔỖ]', 'O', letter)
        letter = re.sub(u'[ơờớợởỡ]', 'ơ', letter)
        letter = re.sub(u'[ƠỜỚỢỞỠ]', 'Ơ', letter)
        letter = re.sub(u'[ìíịỉĩ]', 'i', letter)
        letter = re.sub(u'[ÌÍỊỈĨ]', 'I', letter)
        letter = re.sub(u'[ùúụủũ]', 'u', letter)
        letter = re.sub(u'[ÙÚỤỦŨ]', 'U', letter)
        letter = re.sub(u'[ưừứựửữ]', 'ư', letter)
        letter = re.sub(u'[ƯỪỨỰỬỮ]', 'Ư', letter)
        letter = re.sub(u'[ỳýỵỷỹ]', 'y', letter)
        letter = re.sub(u'[ỲÝỴỶỸ]', 'Y', letter)
        return letter

    def get_accents(self, letter):
        mark_accent, vowel_accent = None, None
        bi_unicode = unicodedata.decomposition(letter).split()

        if bi_unicode[1]=='0302' or (bi_unicode[0] in self.circumflex_unicodes):
            mark_accent = '<6>' # VNI '<CIRCUMFLEX>'
        if bi_unicode[1]=='0306' or (bi_unicode[0] in self.breve_unicodes):
            mark_accent = '<8>' # '<BREVE>'
        if bi_unicode[1]=='0300':
            vowel_accent = '<F>'
        elif bi_unicode[1]=='0301':
            vowel_accent = '<S>'
        elif bi_unicode[1]=='0303':
            vowel_accent = '<X>'
        elif bi_unicode[1]=='0309':
            vowel_accent = '<R>'
        elif bi_unicode[1]=='0323' or (bi_unicode[0] in self.underdot_unicodes):
            vowel_accent = '<J>'

        return mark_accent, vowel_accent

    def flatten_letter(self, letter):
        if letter not in self.accent_letters:
            return letter, None, None
        unaccent_letter = self.get_unaccent(letter)
        mark_accent, vowel_accent = self.get_accents(letter)
        return unaccent_letter, mark_accent, vowel_accent
    
    def flatten_word(self, word):
        flattened_word, mark_accent_word, vowel_accent_word = '', None, None
        for letter in word:
            unaccent_letter, mark_accent, vowel_accent = self.flatten_letter(letter)
            flattened_word += unaccent_letter
            if mark_accent!=None:
                mark_accent_word = mark_accent
            if vowel_accent!=None:
                vowel_accent_word = vowel_accent
        if mark_accent_word!=None:
            flattened_word += mark_accent_word
        if vowel_accent_word!=None:
            flattened_word += vowel_accent_word
        return flattened_word


if __name__ == '__main__':

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    dataset = VNOnDB('./data/VNOnDB/train_word', './data/VNOnDB/train_word.csv', tf)
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
