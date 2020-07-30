from model import ModelTF, ModelRNN, ModelCTC
import pytorch_lightning as pl
import argparse
from config import Config, initialize
from typing import Any
import os
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    # Add trainer options to parser
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('model_name', type=str, choices=['tf', 'rnn', 'ctc'], help='Transformer or RNN or CTC')
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
    elif temp_args.model_name == 'ctc':
        parser = ModelCTC.add_model_specific_args(parser)
        Model = ModelCTC

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        save_top_k=1,
        verbose=True,
        monitor='WER',
        mode='min',
        prefix=''
    )


    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)

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
