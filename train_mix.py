import argparse
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.utils import setup_logger
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver
from ignite.metrics import Accuracy, Loss, MetricsLambda
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torchvision import transforms, models

from dataset import get_data_loader, VNOnDB, RIMES
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss
from model import ResnetFE

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

import numpy as np
import logging
import yaml

# Reproducible
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = f'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(conf_file: str):
    # Read YAML file
    with open(conf_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        return data_loaded

class Model(nn.Module):
    def __init__(self, cnn, vocab, config):
        super().__init__()
        # self.cnn = ResnetFE(version='resnet18')
        self.cnn = ResnetFE(version='resnet50').cnn

        # self.Ic = nn.Linear(512, config['attn_size'])
        self.Ic = nn.Linear(cnn.n_features, config['attn_size'])
        self.Vc = nn.Linear(vocab.size, config['attn_size'])

        ctc_decoder_layer = nn.TransformerEncoderLayer(512, 8)
        self.ctc_decoder = nn.TransformerEncoder(ctc_decoder_layer, 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config['attn_size'], nhead=config['nhead'])
        self.encoder = nn.TransformerEncoder(encoder_layer, 1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config['attn_size'], nhead=config['nhead'])
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)

        self.character_distribution_ctc = nn.Linear(config['attn_size'], vocab.size) # [B,S,V]
        self.character_distribution_ce = nn.Linear(config['attn_size'], vocab.size)

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        '''
        Shapes:
        -------
            images: [B,C,H,W]
            text: [B,T]
        '''
        ctc_images = self.cnn[:-2](images) # [B,C,H,W]
        ctc_pooled = F.adaptive_avg_pool2d(ctc_images, (1, None)) # [B,C,1,W]
        ctc_pooled = ctc_pooled.squeeze(-2).transpose(-1, -2) # [B,S=W,C]
        ctc_pooled = ctc_pooled.transpose(0,1) # [S,B,C]
        ctc_outputs = self.ctc_decoder(ctc_pooled) # [S,B,C]
        ctc_outputs = self.character_distribution_ctc(ctc_outputs.transpose(0,1)) # [B,S,V]

        full_images = self.cnn[-2:](ctc_images) # [B,C,H,W]
        full_images = F.adaptive_avg_pool2d(full_images, (1, None)) # [B,C,1,W]
        full_images = full_images.squeeze(-2).transpose(-1, -2) # [B,S=W,C]
        full_images = self.Ic(full_images) # [B,S,A]

        text = F.one_hot(text, self.Vc.in_features).float().to(text.device) # [B,T,V]
        text = self.Vc(text) # [B,T,A]
        attn_mask = nn.Transformer.generate_square_subsequent_mask(None, text.size(1)).to(text.device) # [B,T,T]

        full_images = full_images.transpose(0,1) # [S,B,A]
        text = text.transpose(0, 1) # [T,B,A]
        ce_outputs = self.decoder(text, full_images, tgt_mask=attn_mask) # [T,B,A]
        ce_outputs = self.character_distribution_ce(ce_outputs.transpose(0,1)) # [B,T,V]
        return ctc_outputs, ce_outputs


class OutputTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.label_tf = StringTransform(vocab, batch_first=True)
        self.ctc_string_tf = CTCStringTransform(vocab, batch_first=False)

    def __call__(self, output):
        return self.ctc_string_tf(output[0]), self.label_tf(output[1])
        # return self.label_tf(output[-3]), self.label_tf(output[-2])


class CTCStringTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.batch_first = batch_first
        self.vocab = vocab

    def __call__(self, tensor: torch.tensor):
        '''
        Convert a Tensor to a list of Strings
        '''
        if not self.batch_first:
            tensor = tensor.transpose(0,1)
        # tensor: [B,T,V]
        tensor = tensor.argmax(-1)
        strs = []
        for sample in tensor.tolist():
            # sample: [T]
            # remove duplicates
            sample = [sample[0]] + [c for i,c in enumerate(sample[1:]) if c != sample[i]]
            # remove 'blank'
            sample = list(filter(lambda i: i != self.vocab.BLANK_IDX, sample))
            # convert to characters
            sample = list(map(self.vocab.int2char, sample))
            strs.append(sample)
        return strs

def main(args):
    alpha = 0.8

    logger = logging.getLogger('MainTraining')
    logger.info('Device = {}'.format(device))

    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        root_config = checkpoint['config']
    else:
        root_config = load_config(args.config_path)
    best_metrics = dict()

    config = root_config['common']

    train_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(config['scale_height']),
        transforms.Grayscale(3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_loader = get_data_loader(config['dataset'],
                                   'trainval' if args.trainval else 'train',
                                   config['batch_size'],
                                   args.num_workers,
                                   train_transform,
                                   args.debug,
                                   flatten_type=config.get('flatten_type', None),
                                   add_blank=True)

    test_transform = transforms.Compose([
        ImageOps.invert,
        ScaleImageByHeight(config['scale_height']),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_loader = get_data_loader(config['dataset'],
                                 'test' if args.trainval else 'val',
                                 config['batch_size'],
                                 args.num_workers,
                                 test_transform,
                                 args.debug,
                                 flatten_type=config.get('flatten_type', None),
                                 add_blank=True)

    if config['dataset'] in ['vnondb', 'vnondb_line']:
        vocab = VNOnDB.vocab
    elif config['dataset'] == 'rimes':
        vocab = RIMES.vocab

    logger.info('Vocab size = {}'.format(vocab.size))

    if config['cnn'] == 'densenet':
        cnn_config = root_config['densenet']
        cnn = DenseNetFE('densenet161', True)
    elif config['cnn'] == 'squeezenet':
        cnn = SqueezeNetFE()
    elif config['cnn'] == 'efficientnet':
        cnn = EfficientNetFE('efficientnet-b1')
    elif config['cnn'] == 'custom':
        cnn = CustomFE(3)
    elif config['cnn'] == 'resnet':
        cnn = ResnetFE('resnet50')
    elif config['cnn'] == 'resnext':
        cnn = ResnextFE('resnext50')
    elif config['cnn'] == 'deformresnet':
        cnn = DeformResnetFE('resnet18')
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))

    if args.model == 'tf':
        model_config = root_config['tf']
        model = Model(cnn, vocab, model_config)
        # model = CTCModelTFEncoder(cnn, vocab, model_config)
    # elif args.model == 's2s':
    #     model_config = root_config['s2s']
    #     model = CTCModelRNN(cnn, vocab, model_config)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    multi_gpus = torch.cuda.device_count() > 1 and args.multi_gpus
    if multi_gpus:
        logger.info("Let's use %d GPUs!", torch.cuda.device_count())
        model = nn.DataParallel(model, dim=0) # batch dim = 0

    model.to(device)
    crossentropy = nn.CrossEntropyLoss().to(device)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-5,
                           weight_decay=0)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=config['n_epochs_decrease_lr'],
        min_lr=config['end_learning_rate'],
        verbose=True)

    if args.resume:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    log_dir = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    log_dir += '_' + args.model
    if args.comment:
        log_dir += '_' + args.comment
    if args.debug:
        log_dir += '_debug'
    log_dir = os.path.join(args.log_root, log_dir)
    tb_logger = TensorboardLogger(log_dir)
    CKPT_DIR = os.path.join(tb_logger.writer.get_logdir(), 'weights')
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    def step_train(engine, batch):
        model.train()
        optimizer.zero_grad()

        imgs, targets = batch.images.to(device), batch.labels.to(device)
        ctc_outputs, ce_outputs = model(imgs, targets[:, :-1]) # [B,S,V], [B,T,V]

        ctc_outputs = F.log_softmax(ctc_outputs, -1) # [B,S,V]
        ctc_outputs_lengths = torch.tensor(ctc_outputs.size(1)).expand(ctc_outputs.size(0))

        packed_ce_outputs = pack_padded_sequence(ce_outputs, batch.lengths - 1, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets[:, 1:], batch.lengths - 1, batch_first=True)[0]

        ctc_loss = ctc(ctc_outputs.transpose(0, 1), targets[:, 1:-1], ctc_outputs_lengths, (batch.lengths - 2))
        crossentropy_loss = crossentropy(packed_ce_outputs, packed_targets)
        loss = alpha * ctc_loss + (1 - alpha) * crossentropy_loss
        loss.backward()
        optimizer.step()

        return ctc_outputs.transpose(0,1), targets[:, 1:-1], {
            'input_lengths': ctc_outputs_lengths,
            'target_lengths': (batch.lengths - 2),
        }, packed_ce_outputs, packed_targets, loss.item()

    @torch.no_grad()
    def step_val(engine, batch):
        model.eval()

        imgs, targets = batch.images.to(device), batch.labels.to(device)
        ctc_outputs, ce_outputs = model(imgs, targets[:, :-1])
        
        ctc_outputs = F.log_softmax(ctc_outputs, -1) # [B,S,V]
        ctc_outputs_lengths = torch.tensor(ctc_outputs.size(1)).expand(ctc_outputs.size(0))

        packed_ce_outputs = pack_padded_sequence(ce_outputs, batch.lengths - 1, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets[:, 1:], batch.lengths - 1, batch_first=True)[0]

        ctc_loss = ctc(ctc_outputs.transpose(0, 1), targets[:, 1:-1], ctc_outputs_lengths, (batch.lengths - 2))
        crossentropy_loss = crossentropy(packed_ce_outputs, packed_targets)
        loss = alpha * ctc_loss + (1 - alpha) * crossentropy_loss

        return ctc_outputs.transpose(0,1), targets[:, 1:-1], {
            'input_lengths': ctc_outputs_lengths,
            'target_lengths': (batch.lengths - 2),
        }, packed_ce_outputs, packed_targets, loss.item()

    trainer = Engine(step_train)
    Running(Loss(ctc, output_transform=lambda output: output[:3]), reset_interval=args.log_interval).attach(trainer, 'CTC')
    Running(Loss(crossentropy, output_transform=lambda output: output[-3:-1]), reset_interval=args.log_interval).attach(trainer, 'CE')
    # Running(src=None, output_transform=lambda output: output[-1], reset_interval=args.log_interval).attach(trainer, 'Loss')

    ProgressBar(ncols=0, ascii=True, position=0).attach(trainer, 'all')
    trainer.logger = setup_logger('Trainer')
    tb_logger.attach(trainer,
                     event_name=Events.ITERATION_COMPLETED,
                     log_handler=OutputHandler(tag='Train', metric_names='all'))

    evaluator = Engine(step_val)
    # Running(Loss(criterion)).attach(evaluator, 'Loss')
    Running(Loss(ctc, output_transform=lambda output: output[:3]), reset_interval=args.log_interval).attach(trainer, 'CTC')
    Running(Loss(crossentropy, output_transform=lambda output: output[-3:-1]), reset_interval=args.log_interval).attach(trainer, 'CE')
    
    Running(CharacterErrorRate(output_transform=OutputTransform(vocab, False))).attach(evaluator, 'CER')
    Running(WordErrorRate(output_transform=OutputTransform(vocab, False))).attach(evaluator, 'WER')
    

    ProgressBar(ncols=0, ascii=True, position=0).attach(evaluator, 'all')
    evaluator.logger = setup_logger('Evaluator')
    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='Validation',
                                               metric_names='all',
                                               global_step_transform=global_step_from_engine(trainer)),
                                               event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        state = evaluator.run(val_loader)
        # is_better = lr_scheduler.is_better(state.metrics['Loss'], lr_scheduler.best)
        # lr_scheduler.step(state.metrics['Loss'])
        is_better = lr_scheduler.is_better(state.metrics['CER'], lr_scheduler.best)
        lr_scheduler.step(state.metrics['CER'])
        to_save = {
            'config': root_config,
            'model': model.state_dict() if not multi_gpus else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trainer': trainer.state_dict(),
        }

        # torch.save(to_save, os.path.join(CKPT_DIR, f'weights_epoch={trainer.state.epoch}_loss={state.metrics["Loss"]:.3f}.pt'))
        torch.save(to_save, os.path.join(CKPT_DIR, f'weights_epoch={trainer.state.epoch}_cer={state.metrics["CER"]:.3f}.pt'))
        if is_better:
            torch.save(to_save, os.path.join(CKPT_DIR, 'BEST.pt'))
            best_metrics.update(state.metrics)

    logger.info('='*60)
    logger.info(model)
    logger.info('='*60)
    logger.info(root_config)
    logger.info('='*60)
    logger.info('Start training..')
    if args.resume:
        trainer.load_state_dict(checkpoint['trainer'])
        trainer.run(train_loader, max_epochs=5 if args.debug else config['max_epochs'], seed=None)
    else:
        trainer.run(train_loader, max_epochs=5 if args.debug else config['max_epochs'], seed=seed)
    print(best_metrics)
    tb_logger.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('config_path', type=str)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug-model', action='store_true', default=False)
    parser.add_argument('--log-root', type=str, default='./runs')
    parser.add_argument('--trainval', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    main(args)
