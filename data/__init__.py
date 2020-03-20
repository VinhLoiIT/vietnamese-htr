import numpy as np
from .dataset import *
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset

def _get_dataset_partition_helper(dataset, partition, transform):
    if dataset not in ['vnondb', 'rimes', 'iam']:
        raise ValueError('Should be: ' + str(['vnondb', 'rimes', 'iam']))

    if partition not in ['train', 'test', 'val', 'trainval']:
        raise ValueError('Should be: ' + str(['train', 'test', 'val', 'trainval']))

    if dataset == 'vnondb':
        if partition == 'test':
            return VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv', transform)
        if partition == 'train':
            return VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', transform)
        if partition == 'val':
            return VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', transform)
        if partition == 'trainval':
            train = VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', transform)
            val = VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', transform)
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

def get_vocab(dataset):
    vocab = Vocab(dataset)
    return vocab

def get_data_loader(dataset, partition, batch_size, num_workers=1, transform=None, vocab=None, debug=False):
    data = _get_dataset_partition_helper(dataset, partition, transform)
    if vocab is None:
        vocab = get_vocab(dataset)

    if debug:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, collate_fn=vocab, num_workers=num_workers,
                            sampler=SubsetRandomSampler(np.random.permutation(min(batch_size * 5, len(data)))))
    else:
        loader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, collate_fn=vocab, num_workers=num_workers)
    return loader
