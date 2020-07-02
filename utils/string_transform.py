import torch
from typing import List


__all__ = ['StringTransform', 'CTCStringTransform']


Char = List[str]


class StringTransform(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tensor: torch.Tensor, lengths: torch.Tensor) -> List[Char]:
        '''
        Convert a Tensor to a list of Strings

        Shapes:
        -------
        - tensor: [B,T]
        - lengths: [B]

        Returns:
        --------
        - List of characters
        '''
        tensor = tensor.cpu()
        lengths = lengths.cpu()

        strs = []
        for i, length in enumerate(lengths.tolist()):
            chars = list(map(self.vocab.int2char, tensor[i, :length].tolist()))
            chars = self.vocab.process_label_invert(chars)
            strs.append(chars)
        return strs


class CTCStringTransform(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tensor: torch.Tensor) -> List[Char]:
        '''
        Convert a Tensor to a list of Strings
        '''
        strs = []
        for sample in tensor.cpu().tolist():
            # sample: [T]
            # remove duplicates
            sample = [sample[0]] + [c for i,c in enumerate(sample[1:]) if c != sample[i]]
            # remove 'blank'
            sample = list(filter(lambda i: i != self.vocab.BLANK_IDX, sample))
            # convert to characters
            sample = list(map(self.vocab.int2char, sample))
            strs.append(sample)
        return strs
