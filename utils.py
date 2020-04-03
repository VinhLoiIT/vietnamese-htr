import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from skimage.feature import hog
from skimage import exposure

class ScaleImageByHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        new_width = int(width * factor)
        new_height = int(height * factor)
        image = image.resize((new_width, new_height), Image.NEAREST)
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
