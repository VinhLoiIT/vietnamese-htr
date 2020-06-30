from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence

from config import Config
from metrics import CharacterErrorRate, Loss, Running, WordErrorRate
from loss import LabelSmoothingCrossEntropy

from .basesystem import BaseSystem
from .utils import CTCStringTransform, StringTransform


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

    def prepare_metric_inputs(self, decoded: List, batch):
        return decoded[0], decoded[1], batch.labels[:, 1:].to(self.device), batch.lengths - 1

    def prepare_train_metrics(self, loss_fn, log_interval: int) -> Dict:
        train_metrics = {
            'Loss': Running(Loss(loss_fn), reset_interval=log_interval)
        }
        return train_metrics

    def prepare_test_metrics(self, vocab, indistinguish: bool) -> Dict:
        string_tf = StringTransform(vocab, batch_first=True)
        out_tf = lambda outputs: (string_tf(outputs[1][0], outputs[1][1]), string_tf(outputs[1][2], outputs[1][3]))
        metrics = {
            'CER': Running(CharacterErrorRate(output_transform=out_tf, is_indistinguish_letter=indistinguish)),
            'WER': Running(WordErrorRate(output_transform=out_tf, is_indistinguish_letter=indistinguish)),
        }
        return metrics

    def prepare_val_metrics(self, vocab, loss_fn) -> Dict:
        string_tf = StringTransform(vocab, batch_first=True)
        out_tf = lambda outputs: (string_tf(outputs[0], outputs[1]), string_tf(outputs[2], outputs[3]))
        metrics = {
            'Loss': Running(Loss(loss_fn, lambda outputs: outputs[0])),
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
        }
        return metrics

    def prepare_loss_function(self, vocab, **kwargs) -> nn.Module:
        if kwargs.get('smoothing', 0) > 0:
            return LabelSmoothingCrossEntropy(kwargs['smoothing'])
        else:
            return nn.CrossEntropyLoss()

    def is_add_blank(self):
        return False

class CEInference():
    def __init__(self, checkpoint, device):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=device)
        self.device = device
        self.config = Config(checkpoint['config'])

        system = CESystem()
        test_data = system.prepare_test_dataset(self.config) # TODO: move vocab to outside of class
        assert test_data is not None
        self.vocab = system.vocab = system.prepare_vocab(self.config)

        self.image_transform = system.prepare_test_image_transform(self.config)

        self.model = system.prepare_model(self.config)
        self.model.to(self.device)
        print(self.model.load_state_dict(checkpoint['model']))
        self.model.eval()
        self.freeze()

        self.output_transform = StringTransform(self.vocab, batch_first=True)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_ = False

    @torch.no_grad()
    def inference(self, images: List[Image.Image]) -> List[str]:
        images = list(map(self.image_transform, images))
        outputs = []
        for image in images:
            image = image.unsqueeze(0)
            outputs = self.model.greedy(image.to(self.device))
            outputs = self.output_transform(outputs)
        return outputs
