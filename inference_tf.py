import argparse
import datetime
import os

import editdistance as ed
import numpy as np
import torch
import tqdm
from torchvision import transforms

from data import get_data_loader, get_vocab, EOS_CHAR
from model import DenseNetFE, Seq2Seq, Transformer
from utils import ScaleImageByHeight

def inference(model, batch, vocab, device):
    imgs, targets, targets_onehot, lengths = batch

    imgs = imgs.to(device)
    targets = targets.to(device)
    targets_onehot = targets_onehot.to(device)

    outputs, _ = model.greedy(imgs, targets_onehot[[0]])

    index = outputs[1:].topk(1, -1)[1] # ignore <start>
    predicts = index.squeeze().transpose(0, 1) # [B, T]
    predicts_str = []
    for predict in predicts:
        s = [vocab.int2char[x.item()] for x in predict]
        try:
            eos_index = s.index(EOS_CHAR) + 1
        except ValueError:
            eos_index = len(s)
        predicts_str.append(s[:eos_index-1]) # ignore <end>

    targets_str = []
    for target in targets.transpose(0, 1).squeeze():
        s = [vocab.int2char[x.item()] for x in target]
        try:
            eos_index = s.index(EOS_CHAR) + 1
        except ValueError:
            eos_index = len(s)
        targets_str.append(s[1:eos_index-1]) #ignore <start< and <end>

    assert len(predicts_str) == len(targets_str)
    for pair in zip(predicts_str, targets_str):
        print(pair)

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    print('Load weight from {}'.format(args.weight))
    checkpoint = torch.load(args.weight, map_location=device)
    config = checkpoint['config']

    cnn = DenseNetFE(config['depth'],
                     config['n_blocks'],
                     config['growth_rate'])

    vocab = get_vocab(config['dataset'])

    if args.model == 'tf':
        model = Transformer(cnn, vocab.vocab_size, config)
    elif args.model == 's2s':
        # model = Seq2Seq(cnn, vocab_size, config['hidden_size'], config['attn_size'])
        pass
    else:
        raise ValueError('model should be "tf" or "s2s"')
    model.to(device)

    model.load_state_dict(checkpoint['model'])

    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        ScaleImageByHeight(config['scale_height']),
        transforms.ToTensor(),
    ])

    test_loader = get_data_loader(config['dataset'], 'test', config['batch_size'],
                                  test_transform, vocab)

    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        inference(model, batch, vocab, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    main(args)
