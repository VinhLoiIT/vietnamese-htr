import argparse
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.utils import setup_logger
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torchvision import transforms, models

from dataset import get_data_loader, VNOnDB
from model import *
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import numpy as np
import logging
import yaml

# Reproducible
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = f'cuda' if torch.cuda.is_available() else 'cpu'

class CTCStringTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.batch_first = batch_first
        self.blank = vocab.BLANK_IDX
        self.int2char = vocab.int2char

    def __call__(self, tensor: torch.tensor):
        '''
        Convert a Tensor to a list of Strings
        '''
        if not self.batch_first:
            tensor = tensor.transpose(0,1)
        # tensor: [B,T]
        strs = []
        for sample in tensor.tolist():
            # sample: [T]
            # remove duplicates
            sample = [sample[0]] + [c for i,c in enumerate(sample[1:]) if c != sample[i]]
            # remove 'blank'
            sample = list(filter(lambda i: i != self.blank, sample))
            # fix index
            sample = list(map(self.int2char, sample))
            strs.append(sample)
        return strs

class OutputTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.label_tf = StringTransform(vocab, batch_first)
        self.output_tf = CTCStringTransform(vocab, batch_first)

    def __call__(self, output):
        return self.output_tf(output[0]), self.label_tf(output[1])

def main(args):
    logger = logging.getLogger('Testing')
    logger.info('Device = {}'.format(device))
    logger.info('Resuming from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    root_config = checkpoint['config']
    best_metrics = dict()

    config = root_config['common']

    image_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(config['scale_height']),
        transforms.Grayscale(3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    loader = get_data_loader(config['dataset'],
                             'test',
                             config['batch_size'],
                             args.num_workers,
                             image_transform,
                             False,
                             flatten_type=config.get('flatten_type', None),
                             add_blank=True) # CTC need add_blank

    if config['dataset'] in ['vnondb', 'vnondb_line']:
        vocab = VNOnDB.vocab

    logger.info('Vocab size = {}'.format(vocab.size))

    if config['cnn'] == 'densenet':
        cnn_config = root_config['densenet']
        cnn = DenseNetFE('densenet161', True)
    elif config['cnn'] == 'squeezenet':
        cnn = SqueezeNetFE()
    elif config['cnn'] == 'efficientnet':
        cnn = EfficientNetFE('efficientnet-b1')
    elif config['cnn'] == 'custom':
        cnn = CustomFE(3)
    elif config['cnn'] == 'resnet':
        cnn = ResnetFE('resnet18')
    elif config['cnn'] == 'resnext':
        cnn = ResnextFE('resnext50')
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))

    if args.model == 'tf':
        model_config = root_config['tf']
        model = CTCModelTFEncoder(cnn, vocab, model_config)
    elif args.model == 's2s':
        model_config = root_config['s2s']
        model = CTCModelRNN(cnn, vocab, model_config)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # beamsearch = model.module.beamsearch if multi_gpus else model.beamsearch
    # greedy = model.module.greedy if multi_gpus else model.greedy

    @torch.no_grad()
    def step_val(engine, batch):
        imgs, targets = batch.images.to(device), batch.labels.to(device)
        outputs = model(imgs) # [B,S,V]
        outputs_lengths = torch.tensor(outputs.size(1)).expand(outputs.size(0))
        return outputs.argmax(-1), batch.labels[:, 1:] # ignore <start>

    evaluator = Engine(step_val)
    Running(CharacterErrorRate(logfile='cer_ctc.txt', output_transform=OutputTransform(vocab, True))).attach(evaluator, 'CER')
    Running(WordErrorRate(logfile='wer_ctc.txt', output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')

    eval_pbar = ProgressBar(ncols=0, ascii=True, position=0)
    eval_pbar.attach(evaluator, 'all')
    evaluator.logger = setup_logger('Evaluator')

    logger.info('='*60)
    logger.info(model)
    logger.info('='*60)
    logger.info(root_config)
    logger.info('='*60)
    logger.info('Start training..')
    evaluator.run(loader, max_epochs=1)
    print('Done')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['tf', 's2s'])
    parser.add_argument('weight', type=str)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()

    main(args)
