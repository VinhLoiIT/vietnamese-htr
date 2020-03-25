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
from model import Seq2Seq, Transformer, DenseNetFE, SqueezeNetFE, EfficientNetFE, CustomFE, ResnetFE
from utils import ScaleImageByHeight, HandcraftFeature
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import logging


# Reproducible
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    print('Load weight from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    root_config = checkpoint['config']
    config = root_config['common']


    image_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(config['scale_height']),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_loader = get_data_loader(config['dataset'], 'val', config['batch_size'],
                                 1,
                                 image_transform, args.debug)
    if args.debug:
        vocab = test_loader.dataset.dataset.vocab
    else:
        vocab = test_loader.dataset.vocab

    if config['cnn'] == 'densenet':
        cnn_config = root_config['densenet']
        cnn = DenseNetFE()
    elif config['cnn'] == 'squeezenet':
        cnn = SqueezeNetFE()
    elif config['cnn'] == 'efficientnet':
        cnn = EfficientNetFE('efficientnet-b1')
    elif config['cnn'] == 'custom':
        cnn = CustomFE(3)
    elif config['cnn'] == 'resnet':
        cnn = ResnetFE()
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))

    if args.model == 'tf':
        model_config = root_config['tf']
        model = Transformer(cnn, vocab.size, model_config)
    elif args.model == 's2s':
        model_config = root_config['s2s']
        model = Seq2Seq(cnn, vocab.size, model_config['hidden_size'], model_config['attn_size'])
    else:
        raise ValueError('model should be "tf" or "s2s"')

    model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.oneshot:
        with torch.no_grad():
            imgs, targets, targets_onehot, lengths = next(iter(test_loader))
            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            outputs, _ = model.greedy(imgs, targets_onehot[[0]].transpose(0,1))
            outputs = outputs.topk(1, -1)[1]
            outputs, targets = outputs.squeeze(-1), targets[1:].transpose(0,1).squeeze(-1)
            outputs = outputs.to('cpu').tolist()
            targets = targets.to('cpu').tolist()
            for sample in zip(outputs, targets):
                print(sample)
            exit(0)

    @torch.no_grad()
    def step_val(engine, batch):
        imgs, targets = batch.images.to(device), batch.labels.to(device)
        outputs, _ = model.greedy(imgs, targets[:, [0]], output_weights=False)
        outputs = outputs.argmax(-1)
        return outputs, targets[:, 1:]

    evaluator = Engine(step_val)
    Running(CharacterErrorRate(vocab)).attach(evaluator, 'CER')
    Running(WordErrorRate(vocab)).attach(evaluator, 'WER')

    eval_pbar = ProgressBar(ncols=0, ascii=True, position=0)
    eval_pbar.attach(evaluator, 'all')
    evaluator.logger = setup_logger('Evaluator')

    @evaluator.on(Events.COMPLETED)
    def finish_eval(engine):
        print('Evaluate Complete. Write down to tensorboard...')
        metrics = {
            'hparam/CER': engine.state.metrics['CER'],
            'hparam/WER': engine.state.metrics['WER'],
        }

    print(config)
    print('='*60)
    print(model)
    print('Start evaluating..')
    evaluator.run(test_loader)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--beamsearch', action='store_true', default=False)
    parser.add_argument('--parition', type=str, choices=['train','val','test'], default='test')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--oneshot', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
