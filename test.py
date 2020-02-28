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
from utils import ScaleImageByHeight, HandcraftFeature
from metrics import CharacterErrorRate, WordErrorRate

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from torch.utils.tensorboard import SummaryWriter

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
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
    model.to(device)

    model.load_state_dict(checkpoint['model'])

    test_transform = transforms.Compose([
        ScaleImageByHeight(config['common']['scale_height']),
        HandcraftFeature() if config['common']['use_handcraft'] else transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_loader = get_data_loader(config['common']['dataset'], args.parition, config['common']['batch_size'],
                                  test_transform, vocab, debug=args.debug)

    if args.oneshot:
        model.eval()
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

    def step_val(engine, batch):
        model.eval()
        with torch.no_grad():
            imgs, targets, targets_onehot, lengths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            outputs, _ = model.greedy(imgs, targets_onehot[[0]].transpose(0,1))
            outputs = outputs.topk(1,-1)[1]

            return outputs, targets[1:].transpose(0,1)

    evaluator = Engine(step_val)
    RunningAverage(CharacterErrorRate(vocab.char2int[EOS_CHAR], batch_first=True)).attach(evaluator, 'running_cer')
    RunningAverage(WordErrorRate(vocab.char2int[EOS_CHAR], batch_first=True)).attach(evaluator, 'running_wer')

    @evaluator.on(Events.STARTED)
    def start_eval(engine):
        print(config)
        print('='*60)
        print(model)
        print('Start evaluating..')

    @evaluator.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_terminal(engine):
        print('Iter {}/{} - CER: {:.3f} WER: {:.3f}'.format(engine.state.iteration, len(test_loader), engine.state.metrics['running_cer'], engine.state.metrics['running_wer']))

    @evaluator.on(Events.COMPLETED)
    def finish_eval(engine):
        print('Evaluate Complete. Write down to tensorboard...')
        print('Iter {}/{} - CER: {:.3f} WER: {:.3f}'.format(engine.state.iteration, len(test_loader), engine.state.metrics['running_cer'], engine.state.metrics['running_wer']))
        metrics = {
            'hparam/CER': engine.state.metrics['running_cer'],
            'hparam/WER': engine.state.metrics['running_wer'],
        }

        with open('result.csv', 'w+') as f:
            print(engine.state.metrics['running_cer'], ',', engine.state.metrics['running_wer'], file=f)

    evaluator.run(test_loader)

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
