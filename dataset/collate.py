import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple, Optional


__all__ = [
    'collate_images', 'collate_text'
]


def collate_images(
    images: List[torch.Tensor],
    pad_value: torch.Tensor,
    max_size: Optional[Tuple[int, int]] = None,
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
    pad_value = pad_value.float()

    collated_images = pad_value.repeat(len(images), max_H, max_W, 1)
    collated_images = collated_images.permute(0,3,1,2)
    padding_masks = torch.ones(len(images), max_H, max_W, dtype=torch.bool)
    for i, image in enumerate(images):
        image_row = image.shape[1]
        image_col = image.shape[2]
        collated_images[i, :, :image_row, :image_col] = image
        padding_masks[i, :image_row, :image_col] = False
    return collated_images, padding_masks


def collate_text(
    text: List[torch.Tensor],
    pad_value: Optional[float] = 0,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Params:
    -------
    - text: list of [T*]
    - pad_value: padding value
    - max_length: max length of padded text. If None, it will be calculated from `text`

    Returns:
    --------
    - collated_text: [B,T]
    - lengths: [B]
    '''
    batch_size = len(text)

    labels = pad_sequence(text, batch_first=True, padding_value=pad_value)
    if max_length is None:
        lengths = torch.tensor([len(label) for label in text])
    else:
        lengths = torch.ones(batch_size).fill_(max_length)

    return labels, lengths
