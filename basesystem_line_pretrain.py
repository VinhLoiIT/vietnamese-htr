import argparse
import collections.abc
import datetime
import logging
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler, TensorboardLogger, global_step_from_engine)
from ignite.engine import Engine, Events
from ignite.metrics import Loss
from ignite.contrib.handlers import ProgressBar
from ignite.utils import setup_logger
from PIL import ImageOps
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import RIMES, RIMESLine, VNOnDB
from dataset.vocab import CollateWrapper
from metrics import CharacterErrorRate, Running, WordErrorRate
from model import *
from utils import ScaleImageByHeight, StringTransform
from basesystem import BaseSystem, MAPPING_NAME

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class BaseSystem:
    def __init__(self):
        self.trainer = Engine(self.step_train)
        self.evaluator = Engine(self.step_val)
        self.tester = Engine(self.step_test)

        for name, engine in {
            'Trainer': self.trainer,
            'Evaluator': self.evaluator,
            'Tester': self.tester
        }.items():
            ProgressBar(ncols=0, ascii=True, position=0).attach(engine, 'all')
            engine.logger = setup_logger(name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = f'cuda' if torch.cuda.is_available() else 'cpu'

        # Reproducible
        self.seed = 0
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train(self, config: Dict):
        if config.get('checkpoint', None) is not None:
            self.logger.info('Load configurations from checkpoint')
            checkpoint = torch.load(config['checkpoint'])
            # self.config = update_dict(config, checkpoint['config'])
            self.config = config
        else:
            checkpoint = None
            self.config = config

        self.logger.info('Load train dataset')
        train_data = self.prepare_train_dataset(self.config)
        assert train_data is not None
        self.logger.info('Load val dataset')
        val_data = self.prepare_val_dataset(self.config)

        self.logger.info('Create train loader')
        train_loader = self.prepare_data_loader(train_data, self.config, True)
        if val_data is not None:
            self.logger.info('Create val loader')
            val_loader = self.prepare_data_loader(val_data, self.config, False)
        else:
            self.logger.info('Do not validate')
            val_loader = None

        self.logger.info('Get vocabulary from dataset')
        self.vocab = self.prepare_vocab(self.config)

        self.logger.info('Create model')
        self.model = self.prepare_model(self.config)
        self.multi_gpus = torch.cuda.device_count() > 1 and self.config['multi_gpus']
        if self.multi_gpus:
            self.logger.info(f'Let\'s use {torch.cuda.device_count()} gpus')
            self.model = nn.DataParallel(model, dim=0) # batch dim = 0
        else:
            self.logger.info(f'Move model to {self.device}')
            self.model.to(self.device)
            
        # Step 3: Init loss function
        self.logger.info('Create loss')
        self.loss_fn = self.prepare_loss_function()
        self.loss_fn.to(self.device)

        # Step 3.5: Init train metrics
        self.logger.info('Create train metrics')
        for name, metric in self.prepare_train_metrics().items():
            metric.attach(self.trainer, name)

        # Step 4: Init optimizers
        self.logger.info('Create optimizer')
        self.optimizer = self.prepare_optimizer(self.model, self.config)

        # Step 5: Init LRScheduler
        self.logger.info('Create learning rate scheduler')
        self.lr_scheduler = self.prepare_lr_scheduler(self.optimizer, self.config)

        # Add validation each epoch
        validation = val_loader is not None
        if validation:
            # self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate, val_loader)
            # Add validation metrics
            self.logger.info('Create validation metrics')
            for name, metric in self.prepare_val_metrics().items():
                metric.attach(self.evaluator, name)

        if checkpoint is not None:
            self.logger.info(f"Resuming from checkpoint")
            model_checkpoint = checkpoint['model']
            drop_keys = [key for key in model_checkpoint.keys() if 'Vc' in key or 'character_distribution' in key]
            for key in drop_keys:
                del model_checkpoint[key]
            self.model.load_state_dict(model_checkpoint, strict=False)

            # freeze all but Vc, character_distribution
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.Vc.weight.requires_grad = True
            self.model.Vc.bias.requires_grad = True
            self.model.character_distribution.weight.requires_grad = True
            self.model.character_distribution.bias.requires_grad = True

            @self.trainer.on(Events.EPOCH_STARTED(once=5))
            def unfreeze(engine):
                self.logger.info('Unfreeze all!!')
                for param in self.model.parameters():
                    param.requires_grad = True

                self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate, val_loader)
            # self.model.load_state_dict(checkpoint['model'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # self.trainer.load_state_dict(checkpoint['trainer'])
            seed = None
        else:
            seed = self.seed

        # Add tensorboard logger
        self.prepare_tensorboard(self.config, validation)

        self.logger.info('Start training')
        self.trainer.run(train_loader, self.config['common']['max_epochs'], seed=seed)
        self.logger.info('Training done')

    def test(self, config: Dict):
        assert config.get('checkpoint', None) is not None
        self.logger.info(f'Load configuration from {config["checkpoint"]}')
        checkpoint = torch.load(config['checkpoint'], map_location=self.device)
        self.config = update_dict(config, checkpoint['config'])

        self.logger.info('Load test dataset')
        test_data = self.prepare_test_dataset(self.config)
        assert test_data is not None

        self.logger.info('Create test loader')
        test_loader = self.prepare_data_loader(test_data, self.config, False)

        self.logger.info('Get vocabulary from dataset')
        self.vocab = self.prepare_vocab(self.config)

        self.logger.info('Create model')
        self.model = self.prepare_model(self.config)
        self.multi_gpus = torch.cuda.device_count() > 1 and self.config['multi_gpus']
        if self.multi_gpus:
            self.logger.info(f'Let\'s use {torch.cuda.device_count()} gpus')
            self.model = nn.DataParallel(model, dim=0) # batch dim = 0
        else:
            self.logger.info(f'Move model to {self.device}')
            self.model.to(self.device)

        self.logger.info('Load weights from checkpoint')
        self.model.load_state_dict(checkpoint['model'])

        self.logger.info('Create test metrics')
        for name, metric in self.prepare_test_metrics().items():
            metric.attach(self.tester, name)

        # Add tensorboard logger
        # self.prepare_tensorboard(self.config)
        self.logger.info('Start testing')
        test_state = self.tester.run(test_loader)
        self.logger.info('Test done')
        self.logger.info(test_state.metrics)

    def prepare_log_dir(self, config) -> str:
        log_dir = '{datetime}_{dataset}_{model}_ce{test}{comment}{debug}'.format(
            datetime=datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'),
            dataset=config['common']['dataset'],
            model=config['common']['model'],
            test='_test' if config.get('test', False) else '',
            comment=f'_{config["comment"]}' if 'comment' in config.keys() else '',
            debug='_debug' if config.get('debug', False) else '',
        )
        log_dir = os.path.join(args.log_root, log_dir)
        return log_dir

    def prepare_tensorboard(self, config, log_val: bool) -> None:
        self.log_dir = self.prepare_log_dir(config)
        self.tb_logger = TensorboardLogger(self.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, 'weights')
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.tb_logger.attach(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED,
            log_handler=OutputHandler(tag='Train', metric_names='all'),
        )

        if log_val:
            tb_val_handler = OutputHandler(
                tag='Validation',
                metric_names='all',
                global_step_transform=global_step_from_engine(self.trainer))
            self.tb_logger.attach(
                self.evaluator,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=tb_val_handler)

    def prepare_model(self, config):
        cnn_name = config['common']['cnn']
        cnn_config = config[cnn_name] or {}
        cnn = MAPPING_NAME[cnn_name](**cnn_config)

        model_name = config['common']['model']
        model_config = config[model_name] or {}
        model = MAPPING_NAME[model_name](cnn, self.vocab, model_config)
        return model

    def prepare_vocab(self, config):
        if config['common']['dataset'] in ['vnondb', 'vnondb_line']:
            vocab = VNOnDB.vocab
        elif config['common']['dataset'] == 'rimes':
            vocab = RIMES.vocab
        elif config['common']['dataset'] == 'rimes_line':
            vocab = RIMESLine.vocab
        return vocab

    def prepare_model_forward_inputs(self, batch):
        return batch.images.to(self.device), batch.labels[:, :-1].to(self.device)

    def prepare_model_decode_input(self, batch):
        return (batch.images.to(self.device), )

    def prepare_loss_inputs(self, outputs, batch):
        lengths = batch.lengths - 1
        targets = batch.labels[:, 1:].to(self.device)
        packed_outputs = pack_padded_sequence(outputs, lengths, True)[0]
        packed_targets = pack_padded_sequence(targets, lengths, True)[0]
        return packed_outputs, packed_targets

    def prepare_metric_inputs(self, decoded, batch):
        return decoded, batch.labels[:, 1:].to(self.device)

    def prepare_train_metrics(self) -> Dict:
        train_metrics = {
            'Loss': Running(Loss(self.loss_fn), reset_interval=args.log_interval)
        }
        return train_metrics

    def prepare_test_metrics(self) -> Dict:
        string_tf = StringTransform(self.vocab, batch_first=True)
        out_tf = lambda outputs: list(map(string_tf, outputs))
        metrics = {
            'CER': Running(CharacterErrorRate(output_transform=out_tf)),
            'WER': Running(WordErrorRate(output_transform=out_tf)),
        }
        return metrics

    def prepare_val_metrics(self) -> Dict:
        loss_fn = self.prepare_loss_function()
        string_tf = StringTransform(self.vocab, batch_first=True)
        out_tf = lambda outputs: list(map(string_tf, outputs))
        metrics = {
            'Loss': Running(Loss(loss_fn, lambda outputs: outputs[0]), reset_interval=args.log_interval),
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
        }
        return metrics

    def step_train(self, trainer: Engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = self.prepare_model_forward_inputs(batch)
        outputs = self.model(*inputs)
        loss_input = self.prepare_loss_inputs(outputs, batch)
        loss = self.loss_fn(*loss_input)
        loss.backward()
        self.optimizer.step()
        return loss_input

    def step_val(self, evaluator: Engine, batch):
        self.model.eval()
        forward_inputs = self.prepare_model_forward_inputs(batch)
        outputs = self.model(*forward_inputs)
        decode_inputs = self.prepare_model_decode_input(batch)
        if self.multi_gpus:
            decoded = self.model.module.greedy(*decode_inputs)
        else:
            decoded = self.model.greedy(*decode_inputs)
        loss_input = self.prepare_loss_inputs(outputs, batch)
        metric_input = self.prepare_metric_inputs(decoded, batch)
        return loss_input, metric_input

    def step_test(self, tester: Engine, batch):
        self.model.eval()
        inputs = self.prepare_model_decode_input(batch)
        if self.multi_gpus:
            decoded = self.model.module.greedy(*inputs)
        else:
            decoded = self.model.greedy(*inputs)
        metric_input = self.prepare_metric_inputs(decoded, batch)
        return metric_input

    def validate(self, trainer: Engine, val_loader: DataLoader) -> None:
        val_state = self.evaluator.run(val_loader)
        is_better = self.lr_scheduler.is_better(
            val_state.metrics['CER'],
            self.lr_scheduler.best)
        self.lr_scheduler.step(val_state.metrics['CER'])
        self.save_checkpoint(os.path.join(self.ckpt_dir, 'weights.pt'))
        if is_better:
            self.save_checkpoint(os.path.join(self.ckpt_dir, 'BEST.pt'))

    def prepare_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def prepare_optimizer(self, model, config) -> optim.Optimizer:
        optim_name = config['common']['optimizer']
        optim_config = config[optim_name]
        return MAPPING_NAME[optim_name](model.parameters(), **optim_config)

    def prepare_lr_scheduler(self, optimizer: optim.Optimizer, config: Dict) -> object:
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            **config['lr_scheduler'])
        return lr_scheduler

    def prepare_train_dataset(self, config) -> Dataset:
        dataset_name = config['common']['dataset']
        dataset_config = config[dataset_name]
        csv = dataset_config['train_csv']
        image_folder = dataset_config['train_image_folder']
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(config['common']['scale_height']),
            transforms.Grayscale(3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        return VNOnDB(image_folder, csv, csv, transform, config['common']['flatten_type'])

    def prepare_val_dataset(self, config) -> Dataset:
        dataset_name = config['common']['dataset']
        dataset_config = config[dataset_name]
        train_csv = dataset_config['train_csv']
        csv = dataset_config['validation_csv']
        image_folder = dataset_config['validation_image_folder']
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(config['common']['scale_height']),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return VNOnDB(image_folder, csv, train_csv, transform, config['common']['flatten_type'])

    def prepare_test_dataset(self, config) -> Dataset:
        dataset_name = config['common']['dataset']
        dataset_config = config[dataset_name]
        train_csv = dataset_config['train_csv']
        csv = dataset_config['test_csv']
        image_folder = dataset_config['test_image_folder']
        transform = transforms.Compose([
            ImageOps.invert,
            ScaleImageByHeight(config['common']['scale_height']),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        return VNOnDB(image_folder, csv, train_csv, transform, config['common']['flatten_type'])

    def prepare_data_loader(self, dataset: str, config: Dict, shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=config['common']['batch_size'],
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=config['num_workers'],
            pin_memory=True,
        )
        return dataloader

    def collate_fn(self, batch) -> CollateWrapper:
        return CollateWrapper(batch)

    def save_checkpoint(self, path: str) -> None:
        if self.multi_gpus:
            model_statedict = self.model.module.state_dict()
        else:
            model_statedict = self.model.state_dict()

        to_save = {
            'config': self.config,
            'model': model_statedict,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'trainer': self.trainer.state_dict(),
        }

        torch.save(to_save, path)

def setup_train(args):
    def load_config(conf_file: str):
        with open(conf_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            return data_loaded

    root_config = load_config(args.base_config) if args.base_config else {}
    override_config = load_config(args.config_path)
    args = vars(args)
    args.pop('func')
    update_dict(root_config, override_config)
    update_dict(root_config, args)
    print('Config:')
    print(root_config)
    system = BaseSystem()
    system.train(root_config)

def setup_test(args):
    config = vars(args)
    config.pop('func')
    system = BaseSystem()
    system.test(config)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
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
    train_parser.add_argument('--base-config', type=str, default='./config/base.yaml')
    train_parser.add_argument('--comment', type=str)
    train_parser.add_argument('--trainval', action='store_true', default=False)
    train_parser.add_argument('-c', '--checkpoint', type=str)

    test_parser = subparser.add_parser('test')
    test_parser.set_defaults(func=setup_test)
    test_parser.add_argument('checkpoint', type=str)

    args = parser.parse_args()
    args.func(args)
