from system import CESystem, CTCSystem
import logging
import argparse
import torch
import yaml
import os
from config import Config
from typing import Dict

def setup_train(args: Dict):
    system = CESystem if args.pop('loss', 'ce') == 'ce' else CTCSystem
    config = Config(args.pop('config_path'), **args)
    # TODO: override config
    system().train(config, args['checkpoint'], args['smoothing'])

def setup_test(args: Dict):
    system = CESystem if args.pop('loss', 'ce') == 'ce' else CTCSystem
    checkpoint = args['checkpoint']
    checkpoint = torch.load(checkpoint,
                            map_location='cpu' if args.pop('cpu') else None)

    if args['config'] is None:
        args['config'] = os.path.join(os.path.dirname(args['checkpoint']), 'config.yaml')
    config = yaml.safe_load(open(args.pop('config')))
    system().test(config,
                  checkpoint,
                  args['validation'],
                  args['beam_width'],
                  args['indistinguish'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('loss', choices=['ctc', 'ce'])
    parser.add_argument('--debug', '-D', action='store_true', default=False)
    parser.add_argument('--debug-model', action='store_true', default=False)
    parser.add_argument('--log-root', type=str, default='./runs')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)

    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train')
    train_parser.set_defaults(func=setup_train)
    train_parser.add_argument('config_path', type=str)
    train_parser.add_argument('--smoothing', type=float, default=0)
    train_parser.add_argument('--comment', type=str, default='')
    train_parser.add_argument('--trainval', action='store_true', default=False)
    train_parser.add_argument('-c', '--checkpoint', type=str)

    test_parser = subparser.add_parser('test')
    test_parser.set_defaults(func=setup_test)
    test_parser.add_argument('checkpoint', type=str)
    test_parser.add_argument('--config', type=str)
    test_parser.add_argument('--beam-width', type=int, default=1)
    test_parser.add_argument('--validation', action='store_true', default=False)
    test_parser.add_argument('--indistinguish', action='store_true', default=False)
    test_parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger('PIL').setLevel(level=logging.INFO)
    args = vars(args)
    func = args.pop('func')
    func(args)
