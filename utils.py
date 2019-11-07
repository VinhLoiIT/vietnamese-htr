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
    return list(alphabets)


eos_char = '<eos>'
alphabets = get_vietnamese_alphabets(is_include_space=False) + [eos_char]

char_to_int = dict((c, i) for i, c in enumerate(alphabets))
int_to_char = dict((i, c) for i, c in enumerate(alphabets))

def encode(characters: list):
    """
    Encode a string to one hot vector using alphabets
    """
    return np.array([[char_to_int[char]] for char in characters])

def to_one_hot(encoded_characters):
    result = np.zeros((len(encoded_characters), len(alphabets)))
    for i, char in enumerate(encoded_characters):
        result[i, char] = 1
    return result


def decode(vectors):
    string = ''.join(int_to_char[np.argmax(vector)] for vector in vectors)
    string = string.replace(eos_char, '')
    return string
