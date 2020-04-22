from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .basesystem import BaseSystem
from metrics import CharacterErrorRate, Loss, Running, WordErrorRate
from utils import CTCStringTransform, StringTransform

__all__ = [
    'CESystem',
]


class CESystem(BaseSystem):
    def __init__(self):
        super().__init__()

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

    def prepare_train_metrics(self, log_interval: int) -> Dict:
        train_metrics = {
            'Loss': Running(Loss(self.loss_fn), reset_interval=log_interval)
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
            'Loss': Running(Loss(loss_fn, lambda outputs: outputs[0])),
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
        }
        return metrics

    def prepare_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def is_add_blank(self):
        return False
