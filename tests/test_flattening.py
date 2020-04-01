import unittest
from dataset.vnondb import *

# class FlatteningTestCase(unittest.TestCase):
#     def setUp(self):
#         self.flattening = Flattening()

#     def test_1(self):
#         string1 = 'quý hóa hoàn khoáng gì gìn đoán đứng lặng HĐND ơn'
#         string2 = []
#         for word in string1.split():
#             word = re.findall(r'\w+', word)[0]
#             flattened_word = self.flattening.flatten_word(word)
#             accent_word = self.flattening.invert(flattened_word)
#             print(f'{word: <{10}} - {flattened_word: <{20}} - {accent_word: <{10}}')
#             string2.append(accent_word)
#         string2 = ' '.join(string2)

#         self.assertEqual(string1, string2) # punctuation->false

#     def test_2(self):
#         df = pd.read_csv(csv, sep='\t', keep_default_na=False, index_col=0)
#         ground_truth = df['label']
#         accent_words = []
#         max_len_flattened = 0
#         for word in ground_truth:
#             flattened_word = flattening.flatten_word(word)
#             if len(re.findall(r'[\w]|<.*?>', flattened_word))>max_len_flattened:
#                 max_len_flattened = len(re.findall(r'[\w]|<.*?>', flattened_word))
#                 print(flattened_word)
#             accent_word = flattening.invert(flattened_word)
#             accent_words.append(accent_word)
#             if word!=accent_word:
#                 print(word, '-', accent_word)
#         sum(ground_truth==accent_words)==len(ground_truth)