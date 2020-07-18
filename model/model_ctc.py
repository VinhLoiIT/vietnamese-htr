from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor

from config import initialize
from dataset import HTRDataset, collate_images, collate_text
from metrics import compute_cer, compute_wer
from utils import CTCStringTransform, ImageTransform, StringTransform

from .aspp import ASPP
from .feature_extractor import ResnetFE


__all__ = [
    'ModelCTC'
]


class ModelCTC(pl.LightningModule):
    '''
    Paper @Are Multidimensional LSTM neccessary for HTR, 2017
    '''

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--beam_width', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--num_layers', type=int, default=1)
        return parser

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters('config')
        self.hparams = config

        self.transform = ImageTransform(augmentation=config.get('augmentation', True),
                                        scale_height=config['dataset']['scale_height'],
                                        min_width=config['dataset']['min_width'])

        self.config = config
        self.beam_width = config['beam_width']

        # define model
        self.cnn = initialize(config['cnn'])
        self.vocab = initialize(config['vocab'], add_blank=True)
        self.loss_fn = nn.CTCLoss(blank=self.vocab.BLANK_IDX)

        output_H = config['dataset']['scale_height'] // (32//2**config['cnn']['args']['droplast'])
        self.blstm = nn.LSTM(input_size=output_H * self.cnn.n_features,
                             hidden_size=config['hidden_size'],
                             num_layers=config['num_layers'],
                             batch_first=True,
                             dropout=config['dropout'],
                             bidirectional=True)
        self.character_distribution = nn.Linear(2*config['hidden_size'], self.vocab.size)
        self.ctc_string_tf = CTCStringTransform(self.vocab)
        self.string_tf = StringTransform(self.vocab)

    def forward(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        ----
        - outputs: [B,T,V]
        '''
        images = self.cnn(images) # [B,C,H',W']
        B, C, H, W = images.shape
        images = images.reshape(B,C*H,W).permute(0,2,1) # [B,T=W,D]
        outputs, _ = self.blstm(images)
        # outputs: [seq_len, batch, num_directions * hidden_size]
        outputs = outputs.transpose(0, 1) # [B,T,D]
        outputs = self.character_distribution(outputs) # [B,T,V]
        return outputs

    def decode(
        self,
        images: torch.Tensor,
        beam_width: int,
        image_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Optional[Tensor], Tensor, Tensor]]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        '''
        if beam_width > 1:
            return self.beamsearch(images, beam_width, image_padding_mask=image_padding_mask)
        else:
            return self.greedy(images, image_padding_mask=image_padding_mask)

    def greedy(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Optional[Tensor], Tensor, Tensor]]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        - weights: weights if `output_weights` is True, else None
        '''
        logits = self(images, image_padding_mask)
        outputs = logits.argmax(-1) # [B,T,V]
        return outputs

    def beamsearch(
        self,
        images: torch.Tensor,
        max_length: int,
        beam_width: int,
        image_padding_mask: Optional[torch.Tensor] = None,
        output_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tuple[Optional[Tensor], Tensor, Tensor]]]:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - outputs: [B,T]
        - lengths: [B]
        '''
        # outputs = self(images, image_padding_mask)
        # outputs = outputs.argmax(-1) # [B,T,V]
        # return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            patience=5,
                                                            min_lr=1.0e-11,
                                                            verbose=True)
        
        return [optimizer], [{
            'scheduler': lr_scheduler, # The LR schduler
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'monitor': self.config['save_by'] # Metric to monitor
        }]

    #########################
    # Training
    #########################

    def training_step(self, batch, batch_idx):
        # prepare forward
        images, labels, image_padding_mask, lengths = batch
        labels, lengths = labels[:, 1:], lengths - 2 # ignore <sos> and <eos>

        # forward
        logits = self(images, image_padding_mask) # [B,T,V]
        log_prob = F.log_softmax(logits, -1)

        predict_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
        loss = self.loss_fn(log_prob, labels, predict_lengths, lengths)

        results = {
            'loss': loss,
            'log': {'Train/Loss': loss.item()},
        }
        return results

    #########################
    # Evaluate
    #########################

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx)
        
    def _shared_eval(self, batch, batch_idx):
        # prepare decode
        images, labels, image_padding_mask, lengths = batch

        # decode
        pred = self.decode(images,
                           self.beam_width,
                           image_padding_mask)
        tgt, tgt_len = labels[:, 1:], lengths - 1

        # convert to strings
        predicts = self.ctc_string_tf(pred)
        groundtruth = self.string_tf(tgt, tgt_len)

        cer_distances, num_chars = compute_cer(predicts, groundtruth, indistinguish=False)
        wer_distances, num_words = compute_wer(predicts, groundtruth, indistinguish=False)

        return {'cer_distances': cer_distances,
                'num_chars': num_chars,
                'wer_distances': wer_distances,
                'num_words': num_words}

    def validation_epoch_end(self, outputs):
        return self._shared_metrics(outputs, 'Validation')

    def test_epoch_end(self, outputs):
        return self._shared_metrics(outputs, 'Test')

    def _shared_metrics(self, outputs, tag: str):
        cer_distances = torch.cat([x['cer_distances'] for x in outputs], dim=0).sum().float()
        num_chars = torch.cat([x['num_chars'] for x in outputs], dim=0).sum()
        wer_distances = torch.cat([x['wer_distances'] for x in outputs], dim=0).sum().float()
        num_words = torch.cat([x['num_words'] for x in outputs], dim=0).sum()
        CER = cer_distances / num_chars.item()
        WER = wer_distances / num_words.item()

        results = {
            'progress_bar': {'CER': CER, 'WER': WER},
            'log': {f'{tag}/CER': CER, f'{tag}/WER': WER, 'step': self.current_epoch},
        }
        return results

    #########################
    # DataLoader
    #########################

    def collate_fn(self, batch):
        batch_size = len(batch)
        batch.sort(key=lambda sample: len(sample[1]), reverse=True)
        image_samples, label_samples = list(zip(*batch))
        images, image_padding_mask = collate_images(image_samples, torch.tensor([0]), None)
        labels, lengths = collate_text(label_samples, 0, None)
        return images, labels, image_padding_mask, lengths

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.prepare_dataset('train', self.vocab, self.config, self.transform.train),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.prepare_dataset('validation', self.vocab, self.config, self.transform.test),
            batch_size=self.config['batch_size'],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        return val_loader

    def test_dataloader(self):
        dataset = self.prepare_dataset('test', self.vocab, self.config, self.transform.test)

        test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        return test_loader

    def prepare_dataset(self, partition: str, vocab, config, image_transform):
        dataset = HTRDataset(vocab=vocab,
                             image_transform=image_transform,
                             **config['dataset'][partition])
        return dataset
