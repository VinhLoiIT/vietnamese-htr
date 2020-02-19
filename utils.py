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
        image = image.resize((new_width, new_height))
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
    
class PaddingWidth(object):
    '''
    Padding width in case of the width is too small
    '''
    def __init__(self, min_width):
        self.min_width = min_width

    def __call__(self, image):
        image_width, image_height = image.size
        width = self.min_width if image_width < self.min_width else image_width
        padded_image = Image.new('RGB', (width, image_height), color='white')
        padded_image.paste(image)
        return padded_image

def accuracy(outputs, targets):
    batch_size = outputs.size(0)
    _, ind = outputs.topk(1, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
