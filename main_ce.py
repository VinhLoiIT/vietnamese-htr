import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pytorch_lightning.metrics.metric import Metric
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from config import Config, initialize
from dataset import CollateWrapper
from metrics import CharacterErrorRate, WordErrorRate
from image_transform import ImageTransform
from utils import StringTransform


class CE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters('config')
        self.hparams = config

        self.vocab = initialize(config['vocab'], add_blank=False)
        self.transform = ImageTransform(augmentation=config.get('augmentation', True),
                                        scale_height=config['dataset']['scale_height'],
                                        min_width=config['dataset']['min_width'])

        cnn = initialize(config['cnn'])
        self.model = initialize(config['model'], cnn, self.vocab)
        self.cer = CharacterErrorRate()
        self.wer = WordErrorRate()
        self.config = config
        self.string_tf = StringTransform(self.vocab)
        self.max_length = config['max_length']
        self.beam_width = config['beam_width']

    def forward(self, images, labels, image_padding_mask, lengths):
        return self.model(images, labels, image_padding_mask, lengths)

    # most cases
    def configure_optimizers(self):
        optimizer = initialize(self.config['optimizer'], self.parameters())
        lr_scheduler = initialize(self.config['lr_scheduler'], optimizer)
        
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
        images = batch.images.to(self.device)
        labels = batch.labels[:, :-1].to(self.device)
        image_padding_mask = batch.image_padding_mask.to(self.device)
        lengths = batch.lengths.to(self.device)

        # forward
        outputs = self(images, labels, image_padding_mask, lengths)

        # prepare loss forward
        lengths = batch.lengths - 1
        targets = batch.labels[:, 1:].to(self.device)
        packed_outputs = pack_padded_sequence(outputs, lengths, True)[0]
        packed_targets = pack_padded_sequence(targets, lengths, True)[0]
        return {'loss': F.cross_entropy(packed_outputs, packed_targets)}

    #########################
    # Validation
    #########################

    def validation_step(self, batch, batch_idx):
        # prepare decode
        images = batch.images.to(self.device)
        image_padding_mask = batch.image_padding_mask.to(self.device)

        # decode
        pred, pred_len = self.model.decode(images, self.max_length,
                                           self.beam_width, image_padding_mask)
        tgt, tgt_len = batch.labels[:, 1:].to(self.device), batch.lengths - 1

        # convert to strings
        predicts = self.string_tf(pred, pred_len)
        groundtruth = self.string_tf(tgt, tgt_len)

        cer_distances = [self.cer.compute_distance(pred, gt) for pred, gt in zip(predicts, groundtruth)]
        wer_distances = [self.wer.compute_distance(pred, gt) for pred, gt in zip(predicts, groundtruth)]

        cer_distances = torch.tensor(cer_distances, dtype=torch.float)
        wer_distances = torch.tensor(wer_distances, dtype=torch.float)

        return {'CER': cer_distances, 'WER': wer_distances}

    def validation_epoch_end(self, outputs):
        CER = torch.stack([x['CER'] for x in outputs]).mean()
        WER = torch.stack([x['WER'] for x in outputs]).mean()
        return {'CER': CER, 'WER': WER}

    #########################
    # Test
    #########################

    def test_step(self, batch, batch_idx):
        # prepare decode
        images = batch.images.to(self.device)
        image_padding_mask = batch.image_padding_mask.to(self.device)

        # decode
        pred, pred_len = self.model.decode(images, self.max_length,
                                           self.beam_width, image_padding_mask)
        tgt, tgt_len = batch.labels[:, 1:].to(self.device), batch.lengths - 1

        # convert to strings
        predicts = self.string_tf(pred, pred_len)
        groundtruth = self.string_tf(tgt, tgt_len)

        cer_distances = [self.cer.compute_distance(pred, gt) for pred, gt in zip(predicts, groundtruth)]
        wer_distances = [self.wer.compute_distance(pred, gt) for pred, gt in zip(predicts, groundtruth)]

        cer_distances = torch.tensor(cer_distances, dtype=torch.float)
        wer_distances = torch.tensor(wer_distances, dtype=torch.float)

        return {'CER': cer_distances, 'WER': wer_distances}

    def test_epoch_end(self, outputs):
        CER = torch.stack([x['CER'] for x in outputs]).mean()
        WER = torch.stack([x['WER'] for x in outputs]).mean()
        return {'CER': CER, 'WER': WER}
    
    #########################
    # DataLoader
    #########################

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.prepare_dataset('train', self.vocab, self.config, self.transform.train),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: CollateWrapper(batch, None),
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.prepare_dataset('validation', self.vocab, self.config, self.transform.test),
            batch_size=self.config['batch_size'],
            collate_fn=lambda batch: CollateWrapper(batch, None),
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
            collate_fn=lambda batch: CollateWrapper(batch, None),
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        return test_loader

    def prepare_dataset(self, partition: str, vocab, config, image_transform) -> Dataset:
        dataset = initialize(config['dataset'],
                             image_transform=image_transform,
                             vocab=vocab,
                             subset=False,
                             **config['dataset'][partition])
        return dataset


if __name__ == "__main__":

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--debug', '-D', action='store_true', default=False)
    common_parser.add_argument('--num-workers', type=int, default=8)
    common_parser.add_argument('--profiler', action='store_true', default=False)
    common_parser.add_argument('--max-length', type=int, default=15)
    common_parser.add_argument('--beam-width', type=int, default=1)

    parser = argparse.ArgumentParser(add_help=True)
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train', parents=[common_parser])
    train_parser.set_defaults(command='train')
    train_parser.add_argument('config_path', type=str)
    train_parser.add_argument('--max-epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--smoothing', type=float, default=0)
    train_parser.add_argument('--comment', type=str, default='')
    train_parser.add_argument('--trainval', action='store_true', default=False)
    train_parser.add_argument('-c', '--checkpoint', type=str, default=None)

    test_parser = subparser.add_parser('test', parents=[common_parser])
    test_parser.set_defaults(command='test')
    test_parser.add_argument('checkpoint', type=str)
    test_parser.add_argument('--hparams', type=str, default=None)
    test_parser.add_argument('--validation', action='store_true', default=False)
    test_parser.add_argument('--indistinguish', action='store_true', default=False)
    test_parser.add_argument('--cpu', action='store_true', default=False)

    cmdargs = parser.parse_args()
    args = vars(cmdargs)

    if args.pop('command') == 'train':
        trainer = pl.Trainer(fast_dev_run=args.pop('debug'),
                             resume_from_checkpoint=args.pop('checkpoint'),
                             profiler=args.pop('profiler'),
                             max_epochs=args['max_epochs'],
                             check_val_every_n_epoch=1,
                             gpus=1)
        config = Config(args.pop('config_path'), **args).config

        model = CE(config)
        trainer.fit(model)
    else:
        model = CE.load_from_checkpoint(checkpoint_path=args.pop('checkpoint'),
                                        hparams_file=args.pop('hparams'))
        trainer = pl.Trainer()
        trainer.test(model,)
