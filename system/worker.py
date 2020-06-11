from typing import Dict, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler, TensorboardLogger, global_step_from_engine)
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.utils import setup_logger
from typing import Mapping, Dict, Callable


class Worker(object):
    def __init__(
        self,
        tb_logger: TensorboardLogger = None
    ):
        self._engine = Engine(self._step)
        self._engine.logger = setup_logger(self.__class__.__name__)
        ProgressBar(ncols=0, ascii=True, position=0).attach(self._engine, 'all')
        self.logger = self._engine.logger
        # Reproducible
        self.seed = 0
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.tb_logger = tb_logger

    def _step(self, engine: Engine, batch):
        pass

    def _attach_metrics(self, metrics):
        for name, metric in metrics.items():
            metric.attach(self._engine, name)


class TrainWorker(Worker):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: Dict,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        loss_input_tf: Callable,
        forward_input_tf: Callable,
        save_metric_best: str,
        checkpoint_dir: str,
        config: Mapping,
        tb_logger: TensorboardLogger = None
    ):
        super().__init__(tb_logger)
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tb_logger = tb_logger
        self.checkpoint_dir = checkpoint_dir
        self.save_metric_best = save_metric_best
        self.config = config
        self.forward_input_tf = forward_input_tf
        self.loss_input_tf = loss_input_tf

        self._attach_metrics(metrics)

    def _step(self, trainer: Engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = self.forward_input_tf(batch)
        outputs = self.model(*inputs)
        loss_input = self.loss_input_tf(outputs, batch)
        loss = self.loss_fn(*loss_input)
        loss.backward()
        self.optimizer.step()
        return loss_input

    def resume(self, data_loader, checkpoint: Mapping):
        self.logger.info('Resuming from checkpoint')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self._engine.load_state_dict(checkpoint['trainer'])
        state = self._engine.run(data_loader, None, seed=None)
        return state

    def train(
        self,
        train_loader: DataLoader,
        max_epochs: int,
        evaluator: 'EvalWorker',
        val_loader: DataLoader = None,
        eval_train: bool = False,
        checkpoint: Mapping = None,
    ):
        self._engine.add_event_handler(Events.EPOCH_COMPLETED,
                                       self.validate,
                                       evaluator,
                                       train_loader if eval_train else None,
                                       val_loader)

        if self.tb_logger:
            self.tb_logger.attach(
                self._engine,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=OutputHandler(tag='Train', metric_names='all'),
            )

        if checkpoint:
            state = self.resume(train_loader, checkpoint)
        else:
            self.logger.info('Start training')
            state = self._engine.run(train_loader, max_epochs, seed=self.seed)
        self.logger.info('Training done. Metrics:')
        self.logger.info(state.metrics)
        return state.metrics

    def validate(
        self,
        engine: Engine,
        evaluator: 'EvalWorker',
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        if train_loader:
            train_metrics: Dict = evaluator.eval(train_loader)
        else:
            train_metrics = None

        if val_loader:
            val_metrics: Dict = evaluator.eval(val_loader)
        else:
            val_metrics = None

        if val_metrics:
            is_better = self.lr_scheduler.is_better(
                val_metrics[self.save_metric_best],
                self.lr_scheduler.best)
            self.lr_scheduler.step(val_metrics[self.save_metric_best])
        elif train_metrics:
            is_better = self.lr_scheduler.is_better(
                train_metrics[self.save_metric_best],
                self.lr_scheduler.best)
            self.lr_scheduler.step(train_metrics[self.save_metric_best])
        else:
            is_better = True

        self.save_checkpoint(os.path.join(self.checkpoint_dir, 'weights.pt'))
        if is_better:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, 'BEST.pt'))

        if self.tb_logger:
            if train_metrics:
                self._log_tb_metrics('Train', engine.state.epoch, train_metrics)
            if val_metrics:
                self._log_tb_metrics('Validation', engine.state.epoch, val_metrics)

    def _log_tb_metrics(self, tag: str, step: int, metrics: Dict):
        for key, value in metrics.items():
            self.tb_logger.writer.add_scalar(f'{tag}/{key}', value, step)

    def save_checkpoint(self, path: str) -> None:
        to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'trainer': self._engine.state_dict(),
        }

        torch.save(to_save, path)


class EvalWorker(Worker):
    def __init__(
        self,
        model: nn.Module,
        metrics: Dict,
        decode_func: Callable,
        decode_input_tf: Callable,
        forward_input_tf: Callable,
        loss_input_tf: Callable,
        metric_input_tf: Callable,
        tb_logger: TensorboardLogger,
    ):
        super().__init__(tb_logger)
        self.model = model
        self.decode_func = decode_func
        self.decode_input_tf = decode_input_tf
        self.forward_input_tf = forward_input_tf
        self.loss_input_tf = loss_input_tf
        self.metric_input_tf = metric_input_tf
        self._attach_metrics(metrics)

    def eval(self, data_loader) -> Dict:
        state = self._engine.run(data_loader)
        return state.metrics

    @torch.no_grad()
    def _step(self, evaluator: Engine, batch):
        self.model.eval()
        forward_inputs = self.forward_input_tf(batch)
        outputs = self.model(*forward_inputs)
        decode_inputs = self.decode_input_tf(batch)
        decoded = self.decode_func(*decode_inputs)
        return self.loss_input_tf(outputs, batch), self.metric_input_tf(decoded, batch)


class TestWorker(EvalWorker):
    def __init__(
        self,
        model: nn.Module,
        metrics: Dict,
        decode_func: Callable,
        decode_input_tf: Callable,
        metric_input_tf: Callable,
        tb_logger: TensorboardLogger,
    ):
        super().__init__(model, metrics, decode_func, decode_input_tf, None, None, metric_input_tf, tb_logger)

    @torch.no_grad()
    def _step(self, evaluator: Engine, batch):
        self.model.eval()
        inputs = self.decode_input_tf(batch)
        decoded = self.decode_func(*inputs)
        return None, self.metric_input_tf(decoded, batch)
