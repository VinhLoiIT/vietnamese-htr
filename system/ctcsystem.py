from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import Config
from metrics import CharacterErrorRate, Loss, Running, WordErrorRate

from .basesystem import BaseSystem
from .utils import CTCStringTransform, StringTransform


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

    def prepare_train_metrics(self, loss_fn, log_interval: int) -> Dict:
        train_metrics = {
            'Loss': Running(Loss(loss_fn,
                                 batch_size=lambda outputs: len(outputs[1])),
                            reset_interval=log_interval)
        }
        return train_metrics

    def prepare_test_metrics(self, vocab, indistinguish: bool) -> Dict:
        string_tf = StringTransform(vocab, batch_first=True)
        ctc_tf = CTCStringTransform(vocab, batch_first=True)
        out_tf = lambda metric_inputs: (ctc_tf(metric_inputs[0]), string_tf(metric_inputs[1]))
        metrics = {
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]),
                                              is_indistinguish_letter=indistinguish)),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]),
                                         is_indistinguish_letter=indistinguish)),
        }
        return metrics

    def prepare_val_metrics(self, vocab, loss_fn) -> Dict:
        string_tf = StringTransform(vocab, batch_first=True)
        ctc_tf = CTCStringTransform(vocab, batch_first=True)
        out_tf = lambda metric_inputs: (ctc_tf(metric_inputs[0]), string_tf(metric_inputs[1]))
        metrics = {
            'Loss': Running(Loss(loss_fn, lambda outputs: outputs[0])),
            'CER': Running(CharacterErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
            'WER': Running(WordErrorRate(output_transform=lambda outputs: out_tf(outputs[1]))),
        }
        return metrics

    def prepare_loss_function(self, vocab, **kwargs) -> nn.Module:
        return nn.CTCLoss(blank=kwargs['vocab'].BLANK_IDX)

    def is_add_blank(self):
        return True

class CTCInference():
    def __init__(self, checkpoint, device):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=device)
        self.device = device
        self.config = Config(checkpoint['config'])

        system = CTCSystem()
        test_data = system.prepare_test_dataset(self.config) # TODO: move vocab to outside of class
        assert test_data is not None
        self.vocab = system.vocab = system.prepare_vocab(self.config)

        self.image_transform = system.prepare_test_image_transform(self.config)

        self.model = system.prepare_model(self.config)
        self.model.to(self.device)
        print(self.model.load_state_dict(checkpoint['model']))
        self.model.eval()
        self.freeze()

        self.output_transform = CTCStringTransform(self.vocab, batch_first=True)

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
