import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple, Union


def collate_images(
    images: List[torch.Tensor],
    pad_value: torch.Tensor,
    max_size: Tuple[int, int] = None,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
    '''
    max_size: (max_H, max_W)
    '''
    if max_size:
        max_H, max_W = max_size
    else:
        max_H = max([image.size(1) for image in images])
        max_W = max([image.size(2) for image in images])

    if pad_value.size(0) == 1:
        pad_value = pad_value.repeat(3)

    collated_images = pad_value.repeat(len(images), max_H, max_W, 1)
    collated_images = collated_images.permute(0,3,1,2)
    masks = torch.ones(len(images), max_H, max_W, dtype=torch.bool)
    for i, image in enumerate(images):
        image_row = image.shape[1]
        image_col = image.shape[2]
        collated_images[i, :, :image_row, :image_col] = image
        masks[i, :image_row, :image_col] = False
    return collated_images, masks


class CollateImageWrapper:
    def __init__(
        self,
        batch,
        pad_value,
        max_size
        ):
        '''
        Shapes:
        -------
        batch: list of
            - image_path: string path of image
            - image: tensor of [C, H, W]
        pad_value: value for padding
        max_size: (max_H, max_W)

        Returns:
        --------
        - images: tensor of [B, C, max_H, max_W]
        - size: tensor of [B, 2]
        '''
        # images: [B, 3, H, W]
        self.paths, images = zip(*batch)
        self.images, self.masks = collate_images(images, pad_value, max_size)

    def pin_memory(self):
        self.images.pin_memory()
        self.masks.pin_memory()
        return self


class CollateWrapper:
    def __init__(
        self,
        batch,
        max_size: Union[None, Tuple[int, int]],
        pad_value = torch.tensor([0.]),
    ):
        '''
        Shapes:
        -------
        batch: list of 2-tuples:
            - image: tensor of [C, H, W]
            - label: tensor of [T*] where T* varies from labels and includes '<start>' and '<end>' at both ends
        pad_value: value for padding
        max_size: (max_H, max_W) or None. If None, it would be the largest size of the batch

        Returns:
        --------
        - images: tensor of [B, C, max_H, max_W]
        - size: tensor of [B, 2]
        - labels: tensor of [B,max_length,1]
        - length: tensor of [B]
        '''

        batch_size = len(batch)
        batch.sort(key=lambda sample: len(sample[1]), reverse=True)
        image_samples, label_samples = list(zip(*batch))

        self.images, self.image_padding_mask = collate_images(image_samples, pad_value, max_size)
        self.lengths = torch.tensor([len(label) for label in label_samples])
        self.labels = pad_sequence(label_samples, batch_first=True)

    def pin_memory(self):
        self.images.pin_memory()
        self.labels.pin_memory()
        return self
