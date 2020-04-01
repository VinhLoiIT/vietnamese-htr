import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from .vocab import CollateWrapper

from .iam import IAM
from .rimes import RIMES
from .vnondb import VNOnDB


def _get_dataset_partition_helper(dataset, partition, transform, flatten_type):
    if dataset not in ['vnondb', 'rimes', 'iam']:
        raise ValueError('Should be: ' + str(['vnondb', 'rimes', 'iam']))

    if partition not in ['train', 'test', 'val', 'trainval']:
        raise ValueError('Should be: ' + str(['train', 'test', 'val', 'trainval']))

    if dataset == 'vnondb':
        if partition == 'test':
            return VNOnDB('./data/VNOnDB/test_word', './data/VNOnDB/test_word.csv', transform, flatten_type)
        if partition == 'train':
            return VNOnDB('./data/VNOnDB/train_word', './data/VNOnDB/train_word.csv', transform, flatten_type)
        if partition == 'val':
            return VNOnDB('./data/VNOnDB/validation_word', './data/VNOnDB/validation_word.csv', transform, flatten_type)
        if partition == 'trainval':
            train = VNOnDB('./data/VNOnDB/train_word', './data/VNOnDB/train_word.csv', transform, flatten_type)
            val = VNOnDB('./data/VNOnDB/validation_word', './data/VNOnDB/validation_word.csv', transform, flatten_type)
            return ConcatDataset([train, val])
        return None
    elif dataset == 'vnondb_line':
        if partition == 'test':
            return VNOnDB('./data/VNOnDB/line/test_line', './data/VNOnDB/line/test_line.csv', transform)
        if partition == 'train':
            return VNOnDB('./data/VNOnDB/line/train_line', './data/VNOnDB/line/train_line.csv', transform)
        if partition == 'val':
            return VNOnDB('./data/VNOnDB/line/validation_line', './data/VNOnDB/line/validation_line.csv', transform)
        if partition == 'trainval':
            train = VNOnDB('./data/VNOnDB/line/train_line', './data/VNOnDB/line/train_line.csv', transform)
            val = VNOnDB('./data/VNOnDB/line/validation_line', './data/VNOnDB/line/validation_line.csv', transform)
            return ConcatDataset([train, val])
        return None
    elif dataset == 'rimes':
        if partition == 'test':
            return RIMES('./data/RIMES/data_test', './data/RIMES/grount_truth_test_icdar2011.txt', transform)
        if partition == 'train':
            return RIMES('./data/RIMES/trainingsnippets_icdar/training_WR', './data/RIMES/groundtruth_training_icdar2011.txt', transform)
        if partition == 'val':
            return RIMES('./data/RIMES/validationsnippets_icdar/testdataset_ICDAR', './data/RIMES/ground_truth_validation_icdar2011.txt', transform)
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

def get_data_loader(dataset, partition, batch_size, num_workers=1, transform=None, debug=False, flatten_type:str=None):
    data = _get_dataset_partition_helper(dataset, partition, transform, flatten_type)
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
