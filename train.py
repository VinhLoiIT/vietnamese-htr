from model import ModelTF, ModelRNN
import pytorch_lightning as pl
import argparse
from config import Config, initialize
from typing import Any


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    # Add trainer options to parser
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('model_name', type=str, choices=['tf', 'rnn'], help='Transformer or RNN')
    parser.add_argument('config_path', type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # Add model options to parser
    Model: Any
    if temp_args.model_name == 'tf':
        parser = ModelTF.add_model_specific_args(parser)
        Model = ModelTF
    elif temp_args.model_name == 'rnn':
        parser = ModelRNN.add_model_specific_args(parser)
        Model = ModelRNN

    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)

    dict_args = vars(args)

    config = Config(dict_args.pop('config_path'), **dict_args).config

    # pl.seed_everything(dict_args['seed'])
    # cnn = initialize(config['cnn'])

    # pl.seed_everything(dict_args['seed'])
    # vocab = initialize(config['vocab'], add_blank=False)

    pl.seed_everything(dict_args['seed'])
    model = Model(config)

    pl.seed_everything(dict_args['seed'])
    trainer.fit(model)
