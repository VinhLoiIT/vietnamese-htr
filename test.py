import argparse
import datetime
import os

import editdistance as ed
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from data import get_data_loader, get_vocab, EOS_CHAR
from model import DenseNetFE, Seq2Seq, Transformer
from utils import ScaleImageByHeight

def inference(model, data_loader, vocab, device):
    model.eval()
    CE = 0
    WE = 0
    total_words = 0
    total_characters = 0
    with torch.no_grad(), tqdm(data_loader) as t:
        for batch_index, batch in enumerate(t):
            imgs, targets, targets_onehot, lengths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            outputs, _ = model.greedy(imgs, targets_onehot[[0]])

            index = outputs.topk(1, -1)[1]
            predicts = index.squeeze(-1).transpose(0, 1) # [B, T]
            predicts_str = []
            for predict in predicts:
                s = [vocab.int2char[x.item()] for x in predict]
                try:
                    eos_index = s.index(EOS_CHAR)
                except ValueError:
                    eos_index = None
                predicts_str.append(s[:eos_index])

            targets_str = []
            for target in targets.transpose(0, 1).squeeze():
                s = [vocab.int2char[x.item()] for x in target]
                try:
                    eos_index = s.index(EOS_CHAR)
                except ValueError:
                    eos_index = None
                targets_str.append(s[1:eos_index])
            assert len(predicts_str) == len(targets_str)

            for i, pair in enumerate(zip(predicts_str, targets_str)):
                CE += ed.distance(pair[0], pair[1])
                WE += 0 if pair[0] == pair[1] else 1
                if pair[0] != pair[1] and args.verbose:
                    tqdm.write('Batch {}, pair {}: {}/{}'.format(batch_index, i, ''.join(pair[0]), ''.join(pair[1])))

            batch_size = len(imgs)
            total_characters += lengths.sum().item() - batch_size*2 # do not count SOS and EOS in total characters
            total_words += batch_size

            t.set_postfix(CE=CE, CER=CE/total_characters, WE=WE, WER=WE/total_words)

    return CE, CE/total_characters, WE, WE/total_words

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

    CE,CER,WE,WER = inference(model, test_loader, vocab, device)
    print('CE = {}, CER = {}, WE = {}, WER = {}'.format(CE, CER, WE, WER))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('weight', type=str, help='Path to weight of model')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--one-shot', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
