import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageOps, Image
from typing import Union

__all__ = [
    'ScaleImageByHeight',
    'ImageTransform'
]


class ScaleImageByHeight(object):
    def __init__(self, target_height, min_width: Union[int, float] = 2.5):
        self.target_height = target_height
        if isinstance(min_width, float):
            self.min_width = int(target_height * min_width)
        elif isinstance(min_width, int):
            self.min_width = min_width
        else:
            raise ValueError('"min_width" should be int or float')

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        scaled_width = int(width * factor)
        resized_image = image.resize((scaled_width, self.target_height), Image.NEAREST)
        image_width = scaled_width if scaled_width > self.min_width else self.min_width
        if resized_image.size[0] > self.min_width:
            image = resized_image.resize((self.min_width, self.target_height))
        elif resized_image.size[0] < self.min_width:
            image = Image.new('L', (self.min_width, self.target_height))
            image.paste(resized_image)
        else:
            image = resized_image
        return image


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
