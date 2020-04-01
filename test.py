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

from dataset import get_data_loader
from model import ModelTF, ModelRNN, DenseNetFE, SqueezeNetFE, EfficientNetFE, CustomFE, ResnetFE
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import logging


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

def main(cfg):
    logger = logging.getLogger('MainTesting')
    logger.info('Device = {}'.format(device))
    logger.info('Loading weight from {}'.format(hydra.utils.to_absolute_path(cfg.weight)))
    checkpoint = torch.load(hydra.utils.to_absolute_path(cfg.weight), map_location=device)
    cfg = checkpoint['config']

    image_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(config['scale_height']),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_loader = get_data_loader(cfg.dataset['name'],
                                 cfg.common['test_partition'],
                                 cfg.common['batch_size'],
                                 cfg.common['num_workers'],
                                 image_transform,
                                 cfg.get('debug', False))

    if cfg.get('debug', False) == True:
        vocab = val_loader.dataset.dataset.vocab
    else:
        vocab = val_loader.dataset.vocab
    logger.info('Vocab size = {}'.format(vocab.size))

    cnn = hydra.utils.instantiate(cfg.cnn)
    if cfg.decoder.name == 'transformer':
        model = ModelTF(cnn, vocab, cfg.decoder)
    elif cfg.decoder.name == 'rnn':
        model = ModelRNN(cnn, vocab, cfg.decoder)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    # multi_gpus = torch.cuda.device_count() > 1 and args.multi_gpus
    # if multi_gpus:
    #     logger.info("Let's use %d GPUs!", torch.cuda.device_count())
    #     model = nn.DataParallel(model, dim=0) # batch dim = 0

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    @torch.no_grad()
    def step_val(engine, batch):
        imgs, targets = batch.images.to(device), batch.labels.to(device)
        outputs = model.greedy(imgs, targets[:, 0], cfg.common['max_length'])
        return outputs, targets[:, 1:]

    evaluator = Engine(step_val)
    Running(CharacterErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'CER')
    Running(WordErrorRate('wer.txt', output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')

    eval_pbar = ProgressBar(ncols=0, ascii=True, position=0)
    eval_pbar.attach(evaluator, 'all')
    evaluator.logger = setup_logger('Evaluator')

    logger.info('='*60)
    logger.info(model)
    logger.info('='*60)
    logger.info(cfg.pretty())
    logger.info('='*60)
    logger.info('Start evaluating..')
    evaluator.run(val_loader)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('weight', type=str, help='Path to weight of model')
    # parser.add_argument('--beamsearch', action='store_true', default=False)
    # parser.add_argument('--parition', type=str, choices=['train','val','test'], default='test')
    # parser.add_argument('--verbose', action='store_true', default=False)
    # parser.add_argument('--gpu-id', type=int, default=0)
    # parser.add_argument('--log-interval', type=int, default=50)
    # parser.add_argument('--debug', action='store_true', default=False)
    # parser.add_argument('--oneshot', action='store_true', default=False)
    # args = parser.parse_args()

    # main(args)
    main()