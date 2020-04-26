import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from skimage.feature import hog
from skimage import exposure
import tqdm
import pandas as pd
import re
import editdistance as ed
from collections import defaultdict, Counter
import glob
import heapq
import itertools
import tqdm
from nltk.util import ngrams, bigrams
import collections

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class ScaleImageByHeight(object):
    def __init__(self, target_height, min_width:int=None):
        self.target_height = target_height
        self.min_width = int(min_width or target_height * 2.5)

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        scaled_width = int(width * factor)
        resized_image = image.resize((scaled_width, self.target_height), Image.NEAREST)
        image_width = scaled_width if scaled_width > self.min_width else self.min_width
        image = Image.new('L', (self.min_width, self.target_height))
        image.paste(resized_image)
        return image
    
class HandcraftFeature(object):
    def __init__(self, orientations=8):
        self.orientations = orientations
    
    def __call__(self, image):
        image_width, image_height = image.size
        handcraft_img = np.ndarray((image_height, image_width, 3))
        handcraft_img[:, :, 0] = np.array(image.convert('L'))
        handcraft_img[:, :, 1] = np.array(image.filter(ImageFilter.FIND_EDGES).convert('L'))
        _, hog_image = hog(image, orientations=self.orientations, visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        handcraft_img[:, :, 2] = np.array(hog_image_rescaled)
        handcraft_img = Image.fromarray(np.uint8(handcraft_img))
        return handcraft_img

class StringTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.batch_first = batch_first
        self.EOS_int = vocab.char2int(vocab.EOS)
        self.vocab = vocab

    def calc_length(self, tensor: torch.tensor):
        '''
        Calculate length of each string ends with EOS
        '''
        lengths = []
        for sample in tensor.tolist():
            try:
                length = sample.index(self.EOS_int)
            except:
                length = len(sample)
            lengths.append(length)
        return lengths

    def __call__(self, tensor: torch.tensor):
        '''
        Convert a Tensor to a list of Strings
        '''
        if not self.batch_first:
            tensor = tensor.transpose(0,1)
        lengths = self.calc_length(tensor)

        strs = []
        for i, length in enumerate(lengths):
            chars = list(map(self.vocab.int2char, tensor[i, :length].tolist()))
            chars = self.vocab.process_label_invert(chars)
            strs.append(chars)
        return strs

class Spell():
    def __init__(self, corpus_folder='data/corpus', corpus_words=None, corpus_biwords=None):
        df = pd.read_csv('./data/VNOnDB/word/test_word.csv', sep='\t', keep_default_na=False, index_col=0)
        gt_test_words = list(df.loc[:, 'label'].astype(str))
        self.dict_words = set(self._words(open('./data/Viet74K.txt').read()))
        for word in gt_test_words:
            self.dict_words.add(word)
            self.dict_words.add(word.lower())

        self.corpus_folder = corpus_folder
        if corpus_words and corpus_biwords:
            self.corpus_words = corpus_words
            self.corpus_biwords = corpus_biwords
        else:
            self.corpus_words, self.corpus_biwords = self._corpus_words()
        self.build_language_model()
    
    def _words(self, text):
        words = []
        for word in re.findall(r'\w+', text.lower()):
            if len(word)>7:
                continue
            words.append(word)
        return words
    
    def _corpus_words(self):
        print('Loading corpus ...')
        corpus_words = Counter()
        corpus_biwords = Counter()
        for fp in tqdm.tqdm(glob.glob(self.corpus_folder+"/*/*.txt")):
            with open(fp, encoding='utf-16', errors='ignore') as f:
                try:
                    data = f.read()
                    words = self._words(data)
                    corpus_words += Counter(words)
                    # corpus_biwords += Counter([bi_word for bi_word in zip(words[:-1], words[1:])])
                    # corpus_biwords += Counter(zip(words, itertools.islice(words, 1, None)))
                    # corpus_biwords += Counter(list(bigrams(words)))
                    corpus_biwords += Counter(list(ngrams(words, 2)))
                except:
                    continue
        print('Corpus loaded!')
        corpus_words = {k:corpus_words[k] for k in corpus_words if corpus_words[k] > 200} # ignore word with freq = 1
        corpus_words = {k: v for k, v in sorted(corpus_words.items(), key=lambda item: item[1], reverse=True)}
        corpus_biwords = {k:corpus_biwords[k] for k in corpus_biwords if corpus_biwords[k] > 5}
        corpus_biwords = {k: v for k, v in sorted(corpus_biwords.items(), key=lambda item: item[1], reverse=True)}

        return corpus_words, corpus_biwords
    
    def _trigrams(self, word, padding=True):
        trigrams = []
        if padding:
            trigrams += [(None, None, word[0]), (word[-1], None, None)]
            if len(word) > 1:
                trigrams += [(None, word[0], word[1]), (word[-2], word[-1], None)]
        trigrams += [(word[i], word[i+1], word[i+2]) for i in range(len(word)-2)]
        return trigrams
    
    def build_language_model(self): ## different with thesis
        self.lm_letter = defaultdict(lambda: defaultdict(lambda: 0))
        for word in self.corpus_words:
            for c1, c2, c3 in self._trigrams(word):
                self.lm_letter[(c1, c2)][c3] += self.corpus_words[word]
        for c1_c2 in self.lm_letter:
            total_count = float(sum(self.lm_letter[c1_c2].values()))
            for c3 in self.lm_letter[c1_c2]:
                self.lm_letter[c1_c2][c3] /= total_count

        self.lm_word = defaultdict(lambda: defaultdict(lambda: 0))
        for biword in self.corpus_biwords:
            word1, word2 = re.findall(r'\w+', biword)
            self.lm_word[word1][word2] += self.corpus_biwords[biword]
        for word1 in self.lm_word:
            total_count = float(sum(self.lm_word[word1].values()))
            for word2 in self.lm_word[word1]:
                self.lm_word[word1][word2] /= total_count
    
    def correction_words(self, predict_words, top1=True):
        '''
        top1==True -> return list letters of word ['t', 'o', 'à', n]
        top1==False -> return list candidate-words (top3) of word ['toàn', 'toán', 'toản']
        '''
        res = []
        for predict_word in predict_words:
            if re.search(r'\s|[,.!-"(/):;%&*\?]', predict_word):
                res.append(predict_word)
                continue

            predict_word = ''.join(predict_word)
            if predict_word in self.dict_words or predict_word.upper() in self.dict_words or predict_word.lower() in self.dict_words\
                or len([i for i in predict_word if i.isdigit()])*3>=len(predict_word): # hypothesis
                res.append(predict_word)
            else:
                candidates = self._levenshtein_candidates(predict_word)

                score = dict()
                for candidate in candidates:
                    score.update({candidate: self._tf_3gram(candidate, predict_word) + self._n_gram_score(candidate)})

                if top1:
                    candidate_word = max(score.keys(), key=lambda k: score[k])
                    if candidate_word.islower() and predict_word[0].isupper(): # hypothesis
                        candidate_word = candidate_word.replace(candidate_word[0], candidate_word[0].upper(), 1)
                    if ed.distance(candidate_word, predict_word) >= len(predict_word)*2/3: # hypothesis
                        candidate_word = predict_word
                    res.append(candidate_word)
                else: # get top 3
                    candidate_word_3s = heapq.nlargest(3, score, key=score.get)
                    res.append(candidate_word_3s)
            if top1:
                res = [[c for c in word] for word in res]
            else:
                res = [[word] if isinstance(word, str) else word for word in res]
        return res

    def correction_lines(self, predict_lines, target_lines=None, paths=None):
        res = []
        if target_lines==None:
            for predict_line in predict_lines:
                predict_line = ''.join(predict_line)

                words = re.findall(r'\w+|[,.!-"(/):;%&*\?]', predict_line) # word | punctuation
                words_with_space = re.findall(r'\w+| |[,.!-"(/):;%&*\?]', predict_line) # with space
                space_indexs = [pos for pos, char in enumerate(words_with_space) if char == ' ']
                
                # Mono-word
                words = self.correction_words(words, top1=False) # [[[a,b,c], [a,b,d], [a,c,b]], [d,e,f,g], [h,i,j],...]

                # Bi-word
                '''
                words = ['làm', 'nhà', ['toanh', 'tình', 'tính'], ['thương', 'thường']]
                    [1] - candidate biwords = ['nhà toanh', 'nhà tình', nhà tính'] -> 'nhà tình'
                        -> words = ['làm', 'nhà', 'tình', ['thương', 'thường']]
                        candidate biwords = ['tình thường', 'tình thương']
                        -> words = ['làm', 'nhà', 'tình', 'thương']
                    *[2] - candidate tri-words = ['nhà toanh thương', 'nhà toanh thường', 'nhà tình thương', 'nhà tình thường'\
                        'nhà tính thương', 'nhà tính thường'] -> 'nhà tình thương'
                        words = ['làm', 'nhà', 'tình', 'thương']
                '''
                for i in range(1, len(words)-1): # !!! suppose words[-1] is only
                    # words[i-1], words[i] is only, not edit
                    if len(words[i-1])==1 and len(words[i])==1:
                        continue
                    # print(f'Edit {words[i-1]} {words[i]} {words[i+1]}')

                    # words[i-1] | words[i] | words[i+1] is list of word
                    cand_triwords = itertools.product(words[i-1], words[i], words[i+1]) # [('nhà', 'toanh', 'thương'), ('nhà', 'toanh', 'thường')...]
                    
                    # Choose best 3-words in cand_triword, suppose correction_words choose right word candidate, different with Thesis-NDK
                    score = dict()
                    for cand in cand_triwords:
                        score.update({cand: self.lm_word[cand[0]][cand[1]] + self.lm_word[cand[1]][cand[2]]})
                    words[i-1], words[i], words[i+1] = ([word] for word in max(score.keys(), key=lambda k: score[k]))
                    # print(f'\tTo {words[i-1]} {words[i]} {words[i+1]}')

                words = [word[0] for word in words]
                # Upper/lower with punctuation
                for i in range(len(words)-1): # [.], [a,c,b]
                    if words[i][0] in ['.', '!', '?'] and words[i+1][0].islower():
                        words[i+1] = words[i+1].replace(words[i+1][0], words[i+1][0].upper(), 1)

                for i in space_indexs:
                    words.insert(i, ' ')
                res.append([letter for word in words for letter in word])
        return res

    def _levenshtein_candidates(self, predict_word):
        candidates = list()
        dist = dict()
        for word in self.dict_words:
            dist.update({word: ed.distance(predict_word, word)})
        min_dist = min(dist.items(), key=lambda x: x[1])[1]
        for key, value in dist.items():
            if value == min_dist:
                candidates.append(key)
        return candidates
    
    # scoring matches using a simple Term Frequency (TF) count
    def _tf_3gram(self, word1, word2):
        tf_count = 0
        word1 = '##'+word1+'##'
        word2 = '##'+word2+'##'
        n_grams1 = [word1[i:i+3] for i in range(len(word1)-2)]
        n_grams2 = [word2[i:i+3] for i in range(len(word2)-2)]
        for n_gram1 in n_grams1:
            for n_gram2 in n_grams2:
                if n_gram1==n_gram2:
                    tf_count += 1
                    break
        return tf_count
    
    def _n_gram_score(self, word): ## different with thesis
        score = 1.0
        for c1, c2, c3 in self._trigrams(word): #padding?
            score *= self.lm_letter[c1, c2][c3]
        score = score**(1/float(len(word)+2))
        return score

class CTCStringTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.batch_first = batch_first
        self.vocab = vocab

    def __call__(self, tensor: torch.tensor):
        '''
        Convert a Tensor to a list of Strings
        '''
        if not self.batch_first:
            tensor = tensor.transpose(0,1)
        # tensor: [B,T]
        strs = []
        for sample in tensor.tolist():
            # sample: [T]
            # remove duplicates
            sample = [sample[0]] + [c for i,c in enumerate(sample[1:]) if c != sample[i]]
            # remove 'blank'
            sample = list(filter(lambda i: i != self.vocab.BLANK_IDX, sample))
            # convert to characters
            sample = list(map(self.vocab.int2char, sample))
            strs.append(sample)
        return strs
