from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from dataset import HTRDataset, collate_images, collate_text
from metrics import compute_cer, compute_wer
from utils import ImageTransform, length_to_padding_mask

__all__ = [
    'ModelCE'
]


class ModelCE(pl.LightningModule):
    '''
    Base class for all models
    '''
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters('config')
        self.hparams = config

        self.transform = ImageTransform(augmentation=config.get('augmentation', True),
                                        scale_height=config['dataset']['scale_height'],
                                        min_width=config['dataset']['min_width'])

        self.config = config
        self.max_length = config['max_length']
        self.beam_width = config['beam_width']

    def embed_image(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - image_padding_mask: [B,H,W]

        Returns:
        --------
        - image_features: [B,S,C']
        '''

    def embed_text(self, text: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
        - text: [B,T]

        Returns:
        --------
        - text: [B,T,V]
        '''

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        label_padding_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - labels: [B,T]
        - image_padding_mask: [B,H,W]
        - label_padding_masks: [B,T]

        Returns:
        ----
        - outputs: [B,T,V]
        '''

    def decode(
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
        - weights: weights if `output_weights` is True, else None
        '''
        if beam_width > 1:
            return self.beamsearch(images, max_length, beam_width, image_padding_mask=image_padding_mask)
        else:
            return self.greedy(images, max_length, image_padding_mask=image_padding_mask)

    def greedy(
        self,
        images: torch.Tensor,
        max_length: int,
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
        - weights: weights if `output_weights` is True, else None
        '''

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
        - weights: weights if `output_weights` is True, else None
        '''

    # most cases
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

        input_labels = labels[:, :-1]
        input_labels_padding_mask = length_to_padding_mask(lengths - 1)

        # forward
        outputs = self(images, input_labels, image_padding_mask, input_labels_padding_mask)

        # prepare loss forward
        targets = labels[:, 1:]
        packed_outputs = pack_padded_sequence(outputs, lengths - 1, True)[0]
        packed_targets = pack_padded_sequence(targets, lengths - 1, True)[0]

        loss = F.cross_entropy(packed_outputs, packed_targets)

        results = {
            'loss': loss,
            'log': {'Train/Loss': loss},
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
        pred, pred_len, _ = self.decode(images,
                                        self.max_length,
                                        self.beam_width,
                                        image_padding_mask,
                                        output_weights=False)
        tgt, tgt_len = labels[:, 1:], lengths - 1

        # convert to strings
        predicts = self.string_tf(pred, pred_len)
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
        # if validation:
        #     dataset = self.prepare_dataset('validation', self.vocab, self.config, self.transform.test)
        # else:
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
