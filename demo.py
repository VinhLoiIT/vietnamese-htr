import argparse
import datetime
import os

import editdistance as ed
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from datasets import get_data_loader, get_vocab, EOS_CHAR, SOS_CHAR
from model import DenseNetFE, Seq2Seq, Transformer
from utils import ScaleImageByHeight, HandcraftFeature
from metrics import CharacterErrorRate, WordErrorRate

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    print('Load weight from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    config = checkpoint['config']

    if config['common']['cnn'] == 'densenet':
        cnn_config = config['densenet']
        cnn = DenseNetFE(cnn_config['depth'],
                         cnn_config['n_blocks'],
                         cnn_config['growth_rate'])

    vocab = get_vocab(config['common']['dataset'])

    if args.model == 'tf':
        model_config = config['tf']
        model = Transformer(cnn, vocab.vocab_size, model_config)
    elif args.model == 's2s':
        model_config = config['s2s']
        model = Seq2Seq(cnn, vocab.vocab_size, model_config['hidden_size'], model_config['attn_size'])
    else:
        raise ValueError('model should be "tf" or "s2s"')
    model.load_state_dict(checkpoint['model'])

    model = nn.DataParallel(model)
    model.to(device)

    test_transform = transforms.Compose([
        ScaleImageByHeight(config['common']['scale_height']),
        HandcraftFeature() if config['common']['use_handcraft'] else transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ##########################
    model.eval()
    with torch.no_grad():
        while True:
            path = input('Enter image path: ')
            if path == '':
                print('Exit')
                break
            image = Image.open(path)
            image = test_transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)

            sos_input = torch.nn.functional.one_hot(torch.tensor(vocab.char2int[SOS_CHAR]), num_classes=vocab.vocab_size)
            sos_input = sos_input.unsqueeze(0).unsqueeze(0) # [B,1,V] where B = 1
            sos_input = sos_input.to(device)
            outputs, weights = model.module.greedy(image, sos_input, output_weights=args.output_weights)
            outputs = outputs.topk(1, -1)[1] # [B,T,1]
            outputs = outputs.to('cpu')
            outputs = outputs.squeeze(0).squeeze(-1).tolist() # remove batch and 1
            outputs = [vocab.int2char[output] for output in outputs]
            outputs = outputs[:outputs.index(EOS_CHAR)+1]
            print(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('image_path', type=str)
    parser.add_argument('--output-weights', action='store_true', default=False)
    parser.add_argument('--beamsearch', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
