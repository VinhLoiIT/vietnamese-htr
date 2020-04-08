import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from .vocab import CollateWrapper

from .iam import IAM
from .rimes import RIMES
from .vnondb import VNOnDB


def _get_dataset_partition_helper(dataset, partition, transform, flatten_type, add_blank):
    if dataset in ['vnondb', 'vnondb_line']:
        level = 'word' if dataset == 'vnondb' else 'line'
        train_csv = f'./data/VNOnDB/{level}/train_{level}.csv'
        test_csv = f'./data/VNOnDB/{level}/test_{level}.csv'
        validation_csv = f'./data/VNOnDB/{level}/validation_{level}.csv'
        test_image_folder = f'./data/VNOnDB/{level}/test_{level}'
        train_image_folder = f'./data/VNOnDB/{level}/train_{level}'
        validation_image_folder = f'./data/VNOnDB/{level}/validation_{level}'

        if partition == 'test':
            return VNOnDB(test_image_folder, test_csv, train_csv, transform, flatten_type, add_blank=add_blank)
        if partition == 'train':
            return VNOnDB(train_image_folder, train_csv, train_csv, transform, flatten_type, add_blank=add_blank)
        if partition == 'val':
            return VNOnDB(validation_image_folder, validation_csv, train_csv, transform, flatten_type, add_blank=add_blank)
        if partition == 'trainval':
            train = VNOnDB(train_image_folder, train_csv, train_csv, transform, flatten_type, add_blank=add_blank)
            val = VNOnDB(validation_image_folder, validation_csv, train_csv, transform, flatten_type, add_blank=add_blank)
            return ConcatDataset([train, val])
        return None
    elif dataset == 'rimes':
        if partition == 'test':
            return RIMES('./data/RIMES/data_test', './data/RIMES/grount_truth_test_icdar2011.txt', transform, add_blank=add_blank)
        if partition == 'train':
            return RIMES('./data/RIMES/trainingsnippets_icdar/training_WR', './data/RIMES/groundtruth_training_icdar2011.txt', transform, add_blank=add_blank)
        if partition == 'val':
            return RIMES('./data/RIMES/validationsnippets_icdar/testdataset_ICDAR', './data/RIMES/ground_truth_validation_icdar2011.txt', transform, add_blank=add_blank)
        return None
    elif dataset == 'iam':
        if partition == 'test':
            return IAM('./data/IAM/splits/test.uttlist', transform)
        if partition == 'train':
            return IAM('./data/IAM/splits/train.uttlist', transform)
        if partition == 'val':
            return IAM('./data/IAM/splits/validation.uttlist', transform)
        return None

    return None

def collate_fn(batch):
    return CollateWrapper(batch)

def get_data_loader(dataset, partition, batch_size, num_workers=1, transform=None, debug=False, flatten_type:str=None, add_blank=False):
    data = _get_dataset_partition_helper(dataset, partition, transform, flatten_type, add_blank)
    shuffle = partition in ['train', 'trainval']

    if debug:
        data = Subset(data, torch.arange(batch_size*5 + batch_size//2).numpy())
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn,
                            num_workers=num_workers,
                            pin_memory=True)
    return loader
