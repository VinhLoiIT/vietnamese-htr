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
from torchvision import transforms

from dataset import get_data_loader, VNOnDB
from model import ModelTF, ModelRNN, DenseNetFE, SqueezeNetFE, EfficientNetFE, CustomFE, ResnetFE, ResnextFE
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import logging
import yaml

# Reproducible
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = f'cuda' if torch.cuda.is_available() else 'cpu'

class OutputTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.tf = StringTransform(vocab, batch_first)

    def __call__(self, output):
        return list(map(self.tf, output))

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
                             flatten_type=config.get('flatten_type', None))

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
        model = ModelTF(cnn, vocab, model_config)
    elif args.model == 's2s':
        model_config = root_config['s2s']
        model = ModelRNN(cnn, vocab, model_config)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    multi_gpus = torch.cuda.device_count() > 1 and args.multi_gpus
    if multi_gpus:
        logger.info("Let's use %d GPUs!", torch.cuda.device_count())
        model = nn.DataParallel(model, dim=0) # batch dim = 0

    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    greedy = model.module.greedy if multi_gpus else model.greedy

    @torch.no_grad()
    def step_val(engine, batch):
        imgs, targets = batch.images.to(device), batch.labels.to(device)
        outputs = greedy(imgs, targets[:, 0], config['max_length'])
        return outputs, targets[:, 1:]

    evaluator = Engine(step_val)
    Running(CharacterErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'CER')
    if config['dataset'] == 'vnondb_line':
        Running(WordErrorRate(logfile='wer.txt', level='line', output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')
    else:
        Running(WordErrorRate(level='word', output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')

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
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()

    main(args)
