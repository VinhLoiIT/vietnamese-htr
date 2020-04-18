import os
import datetime
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from basesystem import BaseSystem
from metrics import CharacterErrorRate, Running, WordErrorRate, Loss
from utils import CTCStringTransform, ScaleImageByHeight, StringTransform


class CTCSystem(BaseSystem):
    def __init__(self):
        super().__init__()

    def prepare_model_forward_inputs(self, batch) -> Union[List, Tuple]:
        return (batch.images.to(self.device), )

    def prepare_model_decode_input(self, batch) -> Union[List, Tuple]:
        return (batch.images.to(self.device), )

    def prepare_loss_inputs(self, outputs, batch):
        targets = batch.labels[:, 1:-1].to(self.device)
        targets_lengths = batch.lengths - 2
        outputs = outputs.transpose(0,1) # [T,B,V]
        outputs_lengths = torch.tensor(outputs.size(0)).expand(outputs.size(1))
        return (outputs, targets, outputs_lengths, targets_lengths)

    def prepare_metric_inputs(self, decoded, batch):
        return decoded, batch.labels[:, 1:].to(self.device)

    def prepare_train_metrics(self, log_interval: int) -> Dict:
        train_metrics = {
            'Loss': Running(Loss(self.loss_fn,
                                 batch_size=lambda outputs: len(outputs[1])),
                            reset_interval=log_interval)
        }
        return train_metrics

    def prepare_test_metrics(self) -> Dict:
        string_tf = StringTransform(self.vocab, batch_first=True)
        ctc_tf = CTCStringTransform(self.vocab, batch_first=True)
        out_tf = lambda metric_inputs: (ctc_tf(metric_inputs[0]), string_tf(metric_inputs[1]))
        metrics = {
            'CER': Running(CharacterErrorRate(output_transform=out_tf)),
            'WER': Running(WordErrorRate(output_transform=out_tf)),
        }
        return metrics

    def prepare_val_metrics(self) -> Dict:
        loss_fn = self.prepare_loss_function()
        string_tf = StringTransform(self.vocab, batch_first=True)
        ctc_tf = CTCStringTransform(self.vocab, batch_first=True)
        out_tf = lambda metric_inputs: (ctc_tf(metric_inputs[0]), string_tf(metric_inputs[1]))
        metrics = {
            'Loss': Running(Loss(loss_fn, lambda outputs: outputs[0])),
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
        }
        return metrics

    def prepare_loss_function(self) -> nn.Module:
        return nn.CTCLoss(blank=self.vocab.BLANK_IDX)

    def is_add_blank(self):
        return True
