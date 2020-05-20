import datetime
import logging
import os
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events
from torch import optim
from torch.utils.data import DataLoader, Dataset

from config import Config, initialize
from dataset import CollateWrapper

from .image_transform import ImageTransform
from .worker import EvalWorker, TestWorker, TrainWorker

__all__ = [
    'BaseSystem',
]


class BaseSystem:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = f'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Running system on {self.device}')

        # Reproducible
        self.seed = 0
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train(self, config: Config, checkpoint: str = None):
        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location=self.device)
            checkpoint_config = checkpoint['config']
        else:
            checkpoint_config = None

        self.logger.info('Config')
        self.logger.info(config)

        self.logger.info('Get vocabulary from dataset')
        vocab = self.prepare_vocab(config)
        self.logger.debug(f'Vocab size = {vocab.size}')
        self.logger.debug(f'Vocab = {vocab.alphabets}')

        self.logger.info('Create image transform')
        transform = ImageTransform(augmentation=config.get('augmentation', True),
                                   scale_height=config['dataset']['scale_height'],
                                   min_width=config['dataset']['min_width'])
        self.logger.debug('Train transform')
        self.logger.debug(transform.train)
        self.logger.debug('Test transform')
        self.logger.debug(transform.test)

        self.logger.info('Create train loader')
        train_loader = DataLoader(
            dataset=self.prepare_dataset('train', vocab, config, transform.train),
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: CollateWrapper(batch),
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.logger.info('Create model')
        model = self.prepare_model(vocab, config).to(self.device)
        self.logger.info(model)

        # Step 3: Init loss function
        self.logger.info('Create loss')
        loss = self.prepare_loss_function(vocab).to(self.device)

        self.logger.info('Create train metrics')
        train_metrics = self.prepare_train_metrics(loss, config['log_interval'])

        self.logger.info('Create optimizer')
        optimizer = self.prepare_optimizer(model, config)

        self.logger.info('Create learning rate scheduler')
        lr_scheduler = self.prepare_lr_scheduler(optimizer, config)

        loss_input_tf = self.prepare_loss_inputs
        forward_input_tf = self.prepare_model_forward_inputs
        save_metric_best = 'CER'
        val_metrics = self.prepare_val_metrics(vocab, loss)
        log_dir = self.prepare_log_dir(config)
        tb_logger = TensorboardLogger(log_dir)
        checkpoint_dir = os.path.join(log_dir, 'weights')
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger.info('Load val loader')
        val_loader = DataLoader(
            dataset=self.prepare_dataset('validation', vocab, config, transform.test),
            batch_size=config['batch_size'],
            collate_fn=lambda batch: CollateWrapper(batch),
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        decode_func = model.greedy
        decode_input_tf = self.prepare_model_decode_input
        loss_input_tf = self.prepare_loss_inputs
        metric_input_tf = self.prepare_metric_inputs

        evaluator = EvalWorker(
            model,
            val_metrics,
            decode_func,
            decode_input_tf,
            forward_input_tf,
            loss_input_tf,
            metric_input_tf,
            tb_logger)

        trainer = TrainWorker(
            model, loss, train_metrics,
            optimizer, lr_scheduler,
            loss_input_tf,
            forward_input_tf, save_metric_best,
            checkpoint_dir, config.config,
            tb_logger
        )

        trainer.train(train_loader, config['max_epochs'], evaluator, val_loader, checkpoint)

    def test(self, checkpoint: Union[Dict, str],
        validation: bool = False,
        beam_width: int = 1,
        indistinguish: bool = False,
    ):
        self.logger.info(f'Load configuration from checkpoint')
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        config = Config(checkpoint['config'])
        self.logger.info('Config')
        self.logger.info(config)

        self.logger.info('Get vocabulary from dataset')
        vocab = self.prepare_vocab(config)

        self.logger.info('Create image transform')
        transform = ImageTransform(augmentation=config.get('augmentation', True),
                                   scale_height=config['dataset']['scale_height'],
                                   min_width=config['dataset']['min_width'])
        self.logger.debug('Test transform')
        self.logger.debug(transform.test)

        self.logger.info('Create test loader')
        if validation:
            dataset = self.prepare_dataset('validation', vocab, config, transform.test)
        else:
            dataset = self.prepare_dataset('test', vocab, config, transform.test)

        test_loader = DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: CollateWrapper(batch),
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.logger.info('Create model')
        config['model']['args']['beam_width'] = beam_width
        model = self.prepare_model(vocab, config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        for param in model.parameters():
            param.requires_grad = False

        self.logger.info('Create test metrics - {}'.format(
            'case insensitive' if indistinguish else 'case sensitive'
        ))
        test_metrics = self.prepare_test_metrics(vocab, indistinguish)

        # log_dir = self.prepare_log_dir(config)
        # tb_logger = TensorboardLogger(log_dir)

        if beam_width > 1:
            self.logger.info(f'Use beam search algorithm with beam_width = {beam_width}')
            decode_func = model.beamsearch
        else:
            self.logger.info('Use greedy search algorithm')
            decode_func = model.greedy
        decode_input_tf = self.prepare_model_decode_input
        metric_input_tf = self.prepare_metric_inputs

        tester = TestWorker(
            model,
            test_metrics,
            decode_func,
            decode_input_tf,
            metric_input_tf,
            None)
            # tb_logger)

        metrics = tester.eval(test_loader)
        self.logger.info('Test done. Metrics:')
        self.logger.info(metrics)

    def prepare_log_dir(self, config) -> str:
        log_dir = '{datetime}_{dataset}_{model}{comment}{debug}'.format(
            datetime=datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'),
            dataset=config['dataset']['name'],
            model=config['model']['name'],
            # test='_test' if config.get('test', False) else '',
            comment=f'_{config["comment"]}' if len(config['comment']) > 0 else '',
            debug='_debug' if config['debug'] else '',
        )
        log_dir = os.path.join(config['log_root'], log_dir)
        return log_dir

    def prepare_model(self, vocab, config: Config):
        cnn = initialize(config['cnn'])
        model = initialize(config['model'], cnn, vocab)
        return model

    def prepare_vocab(self, config):
        vocab = initialize(config['vocab'], add_blank=self.is_add_blank())
        return vocab

    def prepare_model_forward_inputs(self, batch):
        pass

    def prepare_model_decode_input(self, batch):
        return (batch.images.to(self.device), )

    def prepare_loss_inputs(self, outputs, batch):
        pass

    def prepare_metric_inputs(self, decoded, batch):
        return decoded, batch.labels[:, 1:].to(self.device)

    def prepare_test_metrics(self, vocab, indistinguish: bool) -> Dict:
        pass

    def prepare_val_metrics(self, vocab, loss) -> Dict:
        pass

    def prepare_loss_function(self, vocab) -> nn.Module:
        pass

    def prepare_optimizer(self, model, config) -> optim.Optimizer:
        optimizer = initialize(config['optimizer'], model.parameters())
        return optimizer

    def prepare_lr_scheduler(self, optimizer: optim.Optimizer, config: Dict) -> object:
        lr_scheduler = initialize(config['lr_scheduler'], optimizer)
        return lr_scheduler

    def is_add_blank(self):
        pass

    def prepare_dataset(self, partition: str, vocab, config, image_transform) -> Dataset:
        dataset = initialize(config['dataset'],
                             image_transform=image_transform,
                             vocab=vocab,
                             subset=config['debug'],
                             **config['dataset'][partition])
        return dataset
