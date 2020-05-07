import collections.abc
import datetime
import logging
import os
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events
from PIL import ImageOps
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config, initialize
from dataset import *

from .utils import ScaleImageByHeight
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

        self.logger.info('Get vocabulary from dataset')
        vocab = self.prepare_vocab(config)

        self.logger.info('Create train loader')
        train_loader = DataLoader(
            dataset=self.prepare_train_dataset(vocab, config),
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
            dataset=self.prepare_val_dataset(vocab, config),
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
            val_loader,
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
            train_loader, loss_input_tf,
            forward_input_tf, save_metric_best,
            checkpoint_dir, config.config,
            evaluator, tb_logger
        )

        trainer.train(config['max_epochs'], checkpoint)

    def test(self, checkpoint: Union[Dict, str]):
        self.logger.info(f'Load configuration from checkpoint')
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        config = Config(checkpoint['config'])

        self.logger.info('Get vocabulary from dataset')
        vocab = self.prepare_vocab(config)

        self.logger.info('Create test loader')
        test_loader = DataLoader(
            dataset=self.prepare_test_dataset(vocab, config),
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: CollateWrapper(batch),
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.logger.info('Create model')
        model = self.prepare_model(vocab, config).to(self.device)
        model.load_state_dict(checkpoint['model'])

        self.logger.info('Create test metrics')
        test_metrics = self.prepare_test_metrics(vocab)

        # log_dir = self.prepare_log_dir(config)
        # tb_logger = TensorboardLogger(log_dir)

        decode_func = model.greedy
        decode_input_tf = self.prepare_model_decode_input
        metric_input_tf = self.prepare_metric_inputs

        tester = TestWorker(
            model,
            test_loader,
            test_metrics,
            decode_func,
            decode_input_tf,
            metric_input_tf,
            None)
            # tb_logger)

        metrics = tester.eval()
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
        if config['dataset']['name'] in ['vnondb', 'vnondb_line']:
            vocab = VNOnDBVocab(config['dataset']['train']['csv'], self.is_add_blank())
            #TODO: add flattening vocab
        elif config['dataset']['name'] == 'rimes':
            vocab = RIMESVocab(self.is_add_blank())
        elif config['dataset']['name'] == 'rimes_line':
            vocab = RIMESLineVocab(self.is_add_blank())
        elif config['dataset']['name'] == 'cinnamon':
            vocab = CinnamonVocab(config['dataset']['train']['csv'], self.is_add_blank())
        return vocab

    def prepare_model_forward_inputs(self, batch):
        pass

    def prepare_model_decode_input(self, batch):
        return (batch.images.to(self.device), )

    def prepare_loss_inputs(self, outputs, batch):
        pass

    def prepare_metric_inputs(self, decoded, batch):
        return decoded, batch.labels[:, 1:].to(self.device)

    def prepare_test_metrics(self) -> Dict:
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

    def prepare_train_image_transform(self, config):
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(config['scale_height'], config.get('min_width', None)),
            transforms.Grayscale(3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def prepare_test_image_transform(self, config):
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(config['scale_height'], config.get('min_width', None)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform

    def is_add_blank(self):
        pass

    def prepare_train_dataset(self, vocab, config) -> Dataset:
        transform = self.prepare_train_image_transform(config)
        dataset = initialize(config['dataset'],
                             image_transform=transform,
                             vocab=vocab,
                             **config['dataset']['train'])
        return dataset

    def prepare_val_dataset(self, vocab, config) -> Dataset:
        transform = self.prepare_test_image_transform(config)
        dataset = initialize(config['dataset'],
                             image_transform=transform,
                             vocab=vocab,
                             **config['dataset']['validation'])
        return dataset

    def prepare_test_dataset(self, vocab, config) -> Dataset:
        transform = self.prepare_test_image_transform(config)
        dataset = initialize(config['dataset'],
                             image_transform=transform,
                             vocab=vocab,
                             **config['dataset']['test'])
        return dataset

    def collate_fn(self, batch) -> CollateWrapper:
        return CollateWrapper(batch)
