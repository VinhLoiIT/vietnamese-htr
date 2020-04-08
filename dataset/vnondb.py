import os
from collections import Counter
import re
import unicodedata

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .vocab import CollateWrapper, Vocab
from typing import Union, List

class VNOnDBVocab(Vocab):
    def __init__(self, train_csv: str, add_blank: bool):
        self.train_csv = train_csv
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        df = pd.read_csv(self.train_csv, sep='\t', keep_default_na=False, index_col=0)
        return df['label'].astype(str)

class VNOnDBVocabFlatten(VNOnDBVocab):

    def __init__(self, train_csv: str, flattening: str, add_blank: bool):
        if flattening == 'flattening_1':
            self.flattening = Flattening_1()
        elif flattening == 'flattening_2':
            self.flattening = Flattening_2()
        else:
            raise ValueError(f'Unknow flattening type {flattening}, should be "flattening_1" or "flattening_2"')
        super().__init__(train_csv, add_blank)

    def process_label(self, label: List[str]):
        '''
        Preprocess label (if needed), such as flattening out diacritical marks
        '''
        return self.flattening.flatten_word(label)

    def process_label_invert(self, label: List[str]):
        '''
        Invert preprocessed label (if have), such as invert flattening diacritical marks
        '''
        return self.flattening.invert(label)

class Flattening(object):
    def __init__(self):
        self.accent2unicode = {'<6>': '\u0302', '<8>': '\u0306', '<F>': '\u0300', \
                               '<S>': '\u0301', '<R>': '\u0309', '<X>': '\u0303', '<J>': '\u0323'}
        self.circumflex_unicodes = ['00C2', '00E2', '00CA', '00EA', '00D4', '00F4'] # â, Â, Ê, ...
        self.breve_unicodes = ['0102', '0103'] # ă, Ă
        self.underdot_unicodes = ['1EA0', '1EA1', '1EB8', '1EB9', '1ECC', '1ECD']
        self.accent_letters = 'À Á Ả Ã Ạ Â Ầ Ấ Ẩ Ẫ Ậ Ă Ằ Ắ Ẳ Ẵ Ặ à á ả ã ạ â ầ ấ ẩ ẫ ậ ă ằ ắ ẳ ẵ ặ\
        È É Ẻ Ẽ Ẹ Ê Ề Ế Ể Ễ Ệ è é ẻ ẽ ẹ ê ề ế ể ễ ệ\
        Ì Í Ỉ Ĩ Ị ì í ỉ ĩ ị\
        Ò Ó Ỏ Õ Ọ Ô Ồ Ố Ổ Ỗ Ộ Ơ Ờ Ớ Ở Ỡ Ợ ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ\
        Ù Ú Ủ Ũ Ụ Ư Ừ Ứ Ử Ữ Ự ù ú ủ ũ ụ ư ừ ứ ử ữ ự\
        Ỳ Ý Ỷ Ỹ Ỵ ỳ ý ỷ ỹ ỵ'
        self.accent_letters = self.accent_letters.split()
        
    def get_unaccent(self, letter):
        raise NotImplementedError()
        
    def get_accents(self, letter):
        raise NotImplementedError()
    
    def flatten_letter(self, letter):
        flattened_letter = []
        if letter not in self.accent_letters:
            return letter
        unaccent_letter = self.get_unaccent(letter)
        mark_accent, vowel_accent = self.get_accents(letter)
        flattened_letter.append(unaccent_letter)
        if mark_accent != None:
            flattened_letter.append(mark_accent)
        if vowel_accent != None:
            flattened_letter.append(vowel_accent)
        return flattened_letter
    
    
    def flatten_word(self, word):
        '''
        Types:
        ------
            - word: list of accent-letters
            Return:
            - flattened_word: list of unaccent-letters [and <accent-letters> (if any)]
        '''
        flattened_word = []
        for letter in word:
            flattened_letter = self.flatten_letter(letter)
            flattened_word.extend(flattened_letter)
        return flattened_word
    
    def invert(self, flattened_word):
        raise NotImplementedError()

class Flattening_1(Flattening):
    '''
    Flatten without đ, Đ, ơ, Ơ, ư, Ư
    '''
    def __init__(self):
        super().__init__()
        
    def get_unaccent(self, letter):
        letter = letter.encode('utf-8').decode('utf-8')
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

        if bi_unicode[1]=='0302' or (bi_unicode[0] in self.circumflex_unicodes) or letter=='ậ' or letter=='Ậ':
            mark_accent = '<6>' # VNI '<CIRCUMFLEX>'
        elif bi_unicode[1]=='0306' or (bi_unicode[0] in self.breve_unicodes) or letter=='ặ' or letter=='Ặ':
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

    def invert(self, flattened_word):
        '''
        Types:
        ------
            - flattened_word: list of unaccent-letters [and <accent-letters> (if any)]
            Return:
            - accent_word: list of accent-letters
        '''
        accent_word = []
        for letter in flattened_word:
            if (len(letter) == 1) or (len(accent_word) == 0) or (letter not in self.accent2unicode):
                accent_word.append(letter)
            else: # accent
                accent_letter = unicodedata.normalize('NFC', accent_word[-1] + self.accent2unicode[letter])
                accent_word[-1] = accent_letter
        return accent_word

class Flattening_2(Flattening):
    '''
    Flatten with đ, Đ, ơ, Ơ, ư, Ư
    '''
    def __init__(self):
        super().__init__()
#         self.accent2unicode['<7>'] = '\u031B'
        self.accent2unicode.update({'<7>': '\u031B', '<9>': None})
        self._7_unicodes = ['01A0', '01A1', '01AF', '01B0']
        self.accent_letters.extend(['đ', 'Đ'])
        
    def get_unaccent(self, letter):
        letter = letter.encode('utf-8').decode('utf-8')
        letter = re.sub(u'đ', 'd', letter)
        letter = re.sub(u'Đ', 'D', letter)
        return ''.join(c for c in unicodedata.normalize('NFD', letter)\
                       if unicodedata.category(c) != 'Mn')
        
    def get_accents(self, letter):
        mark_accent, vowel_accent = None, None
        bi_unicode = unicodedata.decomposition(letter).split()

        if letter=='đ' or letter=='Đ':
            mark_accent = '<9>'
        elif bi_unicode[1]=='0302' or (bi_unicode[0] in self.circumflex_unicodes) or letter=='ậ' or letter=='Ậ':
            mark_accent = '<6>' # VNI '<CIRCUMFLEX>'
        elif bi_unicode[1]=='0306' or (bi_unicode[0] in self.breve_unicodes) or letter=='ặ' or letter=='Ặ':
            mark_accent = '<8>' # '<BREVE>'
        elif bi_unicode[1]=='031B' or (bi_unicode[0] in self._7_unicodes):
            mark_accent = '<7>'
            
        if letter=='đ' or letter=='Đ':
            vowel_accent = None
        elif bi_unicode[1]=='0300':
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
    
    def invert(self, flattened_word):
        '''
        Types:
        ------
            - flattened_word: list of unaccent-letters [and <accent-letters> (if any)]
            Return:
            - accent_word: list of accent-letters
        '''
        accent_word = []
        for letter in flattened_word:
            if (len(letter) == 1) or (len(accent_word) == 0) or (letter not in self.accent2unicode):
                accent_word.append(letter)
            else: # accent
                if letter == '<9>':
                    if accent_word[-1] in ['d', 'D']:
                        accent_letter = ('đ' if accent_word[-1]=='d' else 'Đ')
                        accent_word[-1] = accent_letter
                    else:
                        accent_word.append(letter)
                else:
                    accent_letter = unicodedata.normalize('NFC', accent_word[-1] + self.accent2unicode[letter])
                    accent_word[-1] = accent_letter
        return accent_word

class VNOnDB(Dataset):

    vocab : VNOnDBVocab = None

    def __init__(self,
        image_folder: str,
        csv: str,
        train_csv: str=None,
        image_transform=None,
        flatten_type: str=None,
        add_blank: bool=False,
    ):
        if VNOnDB.vocab is None:
            if flatten_type is not None:
                VNOnDB.vocab = VNOnDBVocabFlatten(train_csv, flatten_type, add_blank)
            else:
                VNOnDB.vocab = VNOnDBVocab(train_csv, add_blank)
        self.image_transform = image_transform

        self.df = pd.read_csv(csv, sep='\t', keep_default_na=False, index_col=0)
        self.df['id'] = self.df['id'].apply(lambda id: os.path.join(image_folder, id+'.png'))
        self.df['label'] = self.df['label'].apply(VNOnDB.vocab.process_label).apply(VNOnDB.vocab.add_signals)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        image = Image.open(image_path).convert('L')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        label = torch.tensor(list(map(VNOnDB.vocab.char2int, self.df['label'][idx])))
            
        return image, label

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
