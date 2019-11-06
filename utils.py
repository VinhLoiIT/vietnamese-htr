import numpy as np

def get_vietnamese_alphabets(is_include_space=False):
    lower_vowels_with_dm = u'áàảãạắằẳãặâấầẩẫậíìỉĩịúùủũụưứừửữựéèẻẽẹêếềểễệóòỏõọơớờởỡợôốồổỗộyýỳỷỹỵ'
    upper_vowels_with_dm = lower_vowels_with_dm.upper()
    lower_without_dm = u'abcdefghijklmnopqrstuvwxyzđ'
    upper_without_dm = lower_without_dm.upper()
    digits = '1234567890'
    
    symbols = '?/*+-!,."\':;#%&()[]'
    if is_include_space:
      symbols = symbols + ' '
      
    alphabets = lower_vowels_with_dm + lower_without_dm + upper_vowels_with_dm + upper_without_dm + digits + symbols
    return alphabets

def encode(string):
    """
    Encode a string to one hot vector using alphabets
    """
    encoded_vectors = []
    for char in string:
        vector = [0]*len(alphabets)
        vector[char_to_int[char]] = 1
        encoded_vectors.append(vector)
    return np.array(encoded_vectors, dtype=int)

def decode( vectors):
    string = ''.join(int_to_char[np.argmax(vector)] for vector in vectors)
    string = string.replace(eos_char, '')
    return string


alphabets = get_vietnamese_alphabets(is_include_space=False)
eos_char = '\n'    

alphabets = alphabets + eos_char
char_to_int = dict((c, i) for i, c in enumerate(alphabets))
int_to_char = dict((i, c) for i, c in enumerate(alphabets))
