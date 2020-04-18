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
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torchvision import transforms

from dataset import get_data_loader, VNOnDB, RIMES
from model import *
from utils import ScaleImageByHeight, StringTransform
from metrics import CharacterErrorRate, WordErrorRate, Running
from losses import FocalLoss

from torch.nn.utils.rnn import pack_padded_sequence
from PIL import ImageOps

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

class OutputTransform(object):
    def __init__(self, vocab, batch_first=True):
        self.tf = StringTransform(vocab, batch_first)

    def __call__(self, output):
        return list(map(self.tf, output[2:]))

def main(args):
    logger = logging.getLogger('MainTraining')
    logger.info('Device = {}'.format(device))

    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        root_config = checkpoint['config']
    else:
        logger.info('Load base config..')
        import collections.abc

        def update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        root_config = load_config('config/base.yaml')
        override_config = load_config(args.config_path)
        update(root_config, override_config)
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
                                   add_blank=False)

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
                                 add_blank=False)

    if config['dataset'] in ['vnondb', 'vnondb_line']:
        vocab = VNOnDB.vocab
    elif config['dataset'] == 'rimes':
        vocab = RIMES.vocab

    logger.info('Vocab size = {}'.format(vocab.size))

    cnn_config = root_config[config['cnn']] or {}
    if config['cnn'] == 'densenet':
        cnn = DenseNetFE(**cnn_config)
    elif config['cnn'] == 'squeezenet':
        cnn = SqueezeNetFE(**cnn_config)
    elif config['cnn'] == 'efficientnet':
        cnn = EfficientNetFE(**cnn_config)
    elif config['cnn'] == 'custom':
        cnn = CustomFE(3)
    elif config['cnn'] == 'resnet':
        cnn = ResnetFE(**cnn_config)
    elif config['cnn'] == 'resnext':
        cnn = ResnextFE(**cnn_config)
    elif config['cnn'] == 'deformresnet':
        cnn = DeformResnetFE('resnet18')
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))

    if args.model == 'tf':
        model_config = root_config['tf']
        if model_config['use_encoder']:
            model = ModelTFEncoder(cnn, vocab, model_config)
        else:
            model = ModelTF(cnn, vocab, model_config)
    elif args.model == 's2s':
        model_config = root_config['s2s']
        model = ModelRNN(cnn, vocab, model_config)
    else:
        raise ValueError('model should be "tf" or "s2s"')

    multi_gpus = torch.cuda.device_count() > 1 and args.multi_gpus
    if multi_gpus:
        logger.info("Let's use %d GPUs!", torch.cuda.device_count())
        model = nn.DataParallel(model, dim=0) # batch dim = 0

    if args.debug_model:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        print(model)
        model.eval()
        dummy_image_input = torch.rand(config['batch_size'], 3, config['scale_height'], config['scale_height'] * 2)
        dummy_target_input = torch.rand(config['batch_size'], config['max_length'], vocab.size)
        dummy_output_train = model(dummy_image_input, dummy_target_input)
        dummy_output_greedy = model.greedy(dummy_image_input, dummy_target_input[:,[0]])
        logger.debug(dummy_output_train.shape)
        logger.debug(dummy_output_greedy.shape)
        logger.info('Ok')
        exit(0)

    model.to(device)
    loss_type = config.get('loss', 'crossentropy')
    if loss_type == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focalloss':
        criterion = FocalLoss(gamma=2, alpha=vocab.class_weight)
    else:
        raise ValueError(f'Unknow loss {loss_type}')
    criterion = criterion.to(device)

    if config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=config['start_learning_rate'],
                                  weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['start_learning_rate'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['start_learning_rate'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    else:
        raise ValueError(f'Unknow optimizer {config["optimizer"]}')
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
    log_dir += f"_{config['dataset']}_{args.model}_ce"
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

        logits = model(imgs, targets[:, :-1])

        packed_logits = pack_padded_sequence(logits, (batch.lengths - 1), batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets[:, 1:], (batch.lengths - 1), batch_first=True)[0]

        loss = criterion(packed_logits, packed_targets)
        loss.backward()
        optimizer.step()

        return packed_logits, packed_targets

    @torch.no_grad()
    def step_val(engine, batch):
        model.eval()

        imgs, targets = batch.images.to(device), batch.labels.to(device)
        logits = model(imgs, targets[:, :-1])
        if multi_gpus:
            outputs = model.module.greedy(imgs)
        else:
            outputs = model.greedy(imgs)

        logits = pack_padded_sequence(logits, (batch.lengths - 1), batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets[:, 1:], (batch.lengths - 1), batch_first=True)[0]

        return logits, packed_targets, outputs, targets[:, 1:]

    trainer = Engine(step_train)
    Running(Loss(criterion), reset_interval=args.log_interval).attach(trainer, 'Loss')

    ProgressBar(ncols=0, ascii=True, position=0).attach(trainer, 'all')
    trainer.logger = setup_logger('Trainer')
    tb_logger.attach(trainer,
                     event_name=Events.ITERATION_COMPLETED,
                     log_handler=OutputHandler(tag='Train', metric_names='all'))

    evaluator = Engine(step_val)
    Running(Loss(criterion, output_transform=lambda output: output[:2])).attach(evaluator, 'Loss')
    Running(CharacterErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'CER')
    Running(WordErrorRate(output_transform=OutputTransform(vocab, True))).attach(evaluator, 'WER')
    

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
        lr_scheduler.step(state.metrics['CER'])

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine: Engine):
        is_better = lr_scheduler.is_better(engine.state.metrics['CER'], lr_scheduler.best)
        lr_scheduler.step(engine.state.metrics['CER'])
        to_save = {
            'config': root_config,
            'model': model.state_dict() if not multi_gpus else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trainer': trainer.state_dict(),
        }

        torch.save(to_save, os.path.join(CKPT_DIR, f'weights.pt'))
        if is_better:
            torch.save(to_save, os.path.join(CKPT_DIR, 'BEST.pt'))
            best_metrics.update(engine.state.metrics)

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
