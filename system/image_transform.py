import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageOps
from .utils import ScaleImageByHeight

class ImageTransform(object):
    def __init__(self, augmentation: bool = False, **kwargs):
        self.train = self.__train(augmentation, **kwargs)
        self.test = self.__test(**kwargs)

    def __train(self, augmentation: bool, **kwargs):
        transform = transforms.Compose([
            ImageOps.invert,
            transforms.RandomRotation(10) if augmentation else nn.Identity(),
            ScaleImageByHeight(kwargs['scale_height'],
                               kwargs['min_width']),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.RandomErasing() if augmentation else nn.Identity(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def __test(self, **kwargs):
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(kwargs['scale_height'],
                               kwargs['min_width']),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform
