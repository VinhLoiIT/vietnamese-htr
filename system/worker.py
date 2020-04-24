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
    def __init__(self,
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
    def __init__(self,
        model: nn.Module,
        loss: nn.Module,
        metrics: Dict,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        data_loader: DataLoader,
        loss_input_tf: Callable,
        forward_input_tf: Callable,
        save_metric_best: str,
        checkpoint_dir: str,
        config: Mapping,
        evaluator: Worker = None,
        tb_logger: TensorboardLogger = None
    ):
        super().__init__(tb_logger)
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.tb_logger = tb_logger
        self.evaluator = evaluator
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

    def resume(self, checkpoint: Mapping):
        self.logger.info('Resuming from checkpoint')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self._engine.load_state_dict(checkpoint['trainer'])
        state = self._engine.run(self.data_loader, None, seed=None)
        return state

    def train(self, max_epochs: int, checkpoint: Mapping=None, debug:bool = False):
        if self.evaluator:
            self._engine.add_event_handler(Events.EPOCH_COMPLETED, self.validate)
        
        if self.tb_logger:
            self.tb_logger.attach(
                self._engine,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=OutputHandler(tag='Train', metric_names='all'),
            )
            if self.evaluator:
                tb_val_handler = OutputHandler(
                    tag='Validation',
                    metric_names='all',
                    global_step_transform=global_step_from_engine(self._engine))
                self.tb_logger.attach(
                    self.evaluator._engine,
                    event_name=Events.EPOCH_COMPLETED,
                    log_handler=tb_val_handler)

        if checkpoint:
            state = self.resume(checkpoint)
        else:
            max_epochs = 5 if debug else max_epochs
            self.logger.info('Start training')
            state = self._engine.run(self.data_loader, max_epochs, seed=self.seed)
        self.logger.info('Training done. Metrics:')
        self.logger.info(state.metrics)

    def validate(self, engine: Engine) -> None:
        val_metrics: Dict = self.evaluator.eval()
        is_better = self.lr_scheduler.is_better(
            val_metrics[self.save_metric_best],
            self.lr_scheduler.best)
        self.lr_scheduler.step(val_metrics[self.save_metric_best])
        self.save_checkpoint(os.path.join(self.checkpoint_dir, 'weights.pt'))
        if is_better:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, 'BEST.pt'))

    def save_checkpoint(self, path: str) -> None:
        to_save = {
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'trainer': self._engine.state_dict(),
        }

        torch.save(to_save, path)


class EvalWorker(Worker):
    def __init__(self,
        model: nn.Module,
        data_loader: DataLoader,
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
        self.data_loader = data_loader
        self.decode_func = decode_func
        self.decode_input_tf = decode_input_tf
        self.forward_input_tf = forward_input_tf
        self.loss_input_tf = loss_input_tf
        self.metric_input_tf = metric_input_tf
        self._attach_metrics(metrics)

    def eval(self) -> Dict:
        state = self._engine.run(self.data_loader)
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
    def __init__(self,
        model: nn.Module,
        data_loader: DataLoader,
        metrics: Dict,
        decode_func: Callable,
        decode_input_tf: Callable,
        metric_input_tf: Callable,
        tb_logger: TensorboardLogger,
    ):
        super().__init__(model, data_loader, metrics, decode_func, decode_input_tf, None, None, metric_input_tf, tb_logger)

    @torch.no_grad()
    def _step(self, evaluator: Engine, batch):
        self.model.eval()
        inputs = self.decode_input_tf(batch)
        decoded = self.decode_func(*inputs)
        return None, self.metric_input_tf(decoded, batch)