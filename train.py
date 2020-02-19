import argparse
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data import get_data_loader, get_vocab, PAD_CHAR
from model import Seq2Seq, Transformer, DenseNetFE
from utils import ScaleImageByHeight, HandcraftFeature

from torch.nn.utils.rnn import pack_padded_sequence

import logging

def main(args):
    config = {
        # Common
        'dataset': 'vnondb', # Should be one of 'vnondb', 'rimes', 'iam'
        'cnn': 'densenet', # maybe other CNN # TODO: future implement CNN
        'optimizer': 'RMSprop', # Should be one of 'adam', 'RMSprop'
        'batch_size': 32,
        'scale_height': 128,
        'attn_size': 256,
        'max_length': 10,
        'use_handcraft': False, # Use handcraft features as input instead of Grayscale(3)

        'n_epochs_decrease_lr': 5,
        'start_learning_rate': 1e-5,  # NOTE: paper start with 1e-8
        'end_learning_rate': 1e-11,
        'max_epochs': 50,
        'weight_decay': 0,

        # Densenet only
        'depth': 4,
        'n_blocks': 3,
        'growth_rate': 96,

        # Seq2Seq only
        'hidden_size': 256,

        # Transformer only
        'use_encoder': False,
        'use_FFNN': False,
        # 'encoder_attn': 'scale_dot_product',
        'encoder_attn': 'additive',
        'decoder_attn': 'additive',
        'encoder_decoder_attn': 'additive',
        'direct_additive': False,
        # 'encoder_decoder_attn': 'scale_dot_product',
        'encoder_nhead': 1, # should divisible by CNN.n_features
        'decoder_nhead': 1, # should divisible by vocab_size
        'encoder_decoder_nhead': 1, # should divisible by attn_size
        'encoder_nlayers': 1,
        'decoder_nlayers': 1,
    }

    best_metrics = dict()

    vocab = get_vocab(config['dataset'])
    logger = logging.getLogger('MainTraining')

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info('Device = {}'.format(device))
    logger.info('Vocab size = {}'.format(vocab.vocab_size))

    if args.resume:
        logger.info('Resuming from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        config = checkpoint['config']

    if config['cnn'] == 'densenet':
        cnn = DenseNetFE(config['depth'],
                        config['n_blocks'],
                        config['growth_rate'])
    else:
        raise ValueError('Unknow CNN {}'.format(config['cnn']))
    if args.model == 'tf':
        model = Transformer(cnn, vocab.vocab_size, config)
    elif args.model == 's2s':
        model = Seq2Seq(cnn, vocab.vocab_size, config['hidden_size'], config['attn_size'])
    else:
        raise ValueError('model should be "tf" or "s2s"')

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        logger.info("Let's use %d GPUs!", torch.cuda.device_count())
        model = nn.DataParallel(model, dim=1) # batch dim = 1

    if args.debug_model:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        print(model)
        model.eval()
        dummy_image_input = torch.rand(config['batch_size'], 3, config['scale_height'], config['scale_height'] * 2)
        dummy_target_input = torch.rand(config['max_length'], config['batch_size'], vocab.vocab_size)
        dummy_output_train = model(dummy_image_input, dummy_target_input)
        dummy_output_greedy, _ = model.greedy(dummy_image_input, dummy_target_input[[0]])
        logger.debug(dummy_output_train.shape)
        logger.debug(dummy_output_greedy.shape)
        logger.info('Ok')
        exit(0)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=config['start_learning_rate'],
                                  weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['start_learning_rate'],
                               weight_decay=config['weight_decay'])
    else:
        raise ValueError('Unknow optimizer {}'.format(config['optimizer']))
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=config['n_epochs_decrease_lr'],
        min_lr=config['end_learning_rate'],
        verbose=True)

    if args.resume:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        reduce_lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    image_transform = transforms.Compose([
        ScaleImageByHeight(config['scale_height']),
        HandcraftFeature() if config['use_handcraft'] else transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    train_loader = get_data_loader(config['dataset'], 'train', config['batch_size'],
                                   image_transform, vocab, args.debug)

    val_loader = get_data_loader(config['dataset'], 'val', config['batch_size'],
                                 image_transform, vocab, args.debug)

    log_dir = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    log_dir += '_' + args.model
    if args.comment:
        log_dir += '_' + args.comment
    if args.debug:
        log_dir += '_debug'
    log_dir = os.path.join(args.log_root, log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    CKPT_DIR = os.path.join(writer.get_logdir(), 'weights')
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    def step_train(engine, batch):
        model.train()
        optimizer.zero_grad()

        imgs, targets, targets_onehot, lengths = batch

        imgs = imgs.to(device)
        targets = targets.to(device)
        targets_onehot = targets_onehot.to(device)

        outputs = model(imgs, targets_onehot[:-1])

        packed_outputs = pack_padded_sequence(outputs, (lengths - 1).squeeze(-1))[0]
        packed_targets = pack_padded_sequence(targets[1:].squeeze(-1), (lengths - 1).squeeze(-1))[0]

        loss = criterion(packed_outputs, packed_targets)
        loss.backward()
        optimizer.step()

        return packed_outputs, packed_targets

    def step_val(engine, batch):
        model.eval()

        with torch.no_grad():
            imgs, targets, targets_onehot, lengths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            outputs, _ = model.greedy(imgs, targets_onehot[[0]])

            packed_outputs = pack_padded_sequence(outputs, (lengths - 1).squeeze(-1))[0]
            packed_targets = pack_padded_sequence(targets[1:].squeeze(-1), (lengths - 1).squeeze(-1))[0]

            return packed_outputs, packed_targets

    trainer = Engine(step_train)
    evaluator = Engine(step_val)

    RunningAverage(Loss(criterion)).attach(trainer, 'running_train_loss')
    RunningAverage(Accuracy()).attach(trainer, 'running_train_acc')
    RunningAverage(Loss(criterion)).attach(evaluator, 'running_val_loss')
    RunningAverage(Accuracy()).attach(evaluator, 'running_val_acc')

    training_timer = Timer(average=True).attach(trainer)
    epoch_train_timer = Timer(True).attach(trainer,
                                           start=Events.EPOCH_STARTED,
                                           pause=Events.EPOCH_COMPLETED,
                                           step=Events.EPOCH_COMPLETED)
    batch_train_timer = Timer(True).attach(trainer)

    validate_timer = Timer(average=True).attach(evaluator)

    @trainer.on(Events.STARTED)
    def start_training(engine):
        logger.info('='*60)
        logger.info(config)
        logger.info('='*60)
        logger.info(model)
        logger.info('='*60)
        logger.info('Start training..')

    @trainer.on(Events.COMPLETED)
    def end_training(engine):
        logger.info('Training completed!!')
        logger.info('Total training time: {:3.2f}s'.format(training_timer.value()))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_tensorboard(engine):
        state = engine.state
        writer.add_scalar("Train/Loss", state.metrics['running_train_loss'], state.iteration)
        writer.add_scalar("Train/Accuracy", state.metrics['running_train_acc'], state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_terminal(engine):
        batch_train_timer.pause()
        batch_train_timer.step()
        logger.info("Train - Epoch: {} - Iter {} - Accuracy: {:.3f} Loss: {:.3f} Avg Time: {:.2f}"
              .format(engine.state.epoch, engine.state.iteration,
                      engine.state.metrics['running_train_acc'],
                      engine.state.metrics['running_train_loss'],
                      batch_train_timer.value()))
        batch_train_timer.resume()

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        writer.add_scalar("Train/Time", epoch_train_timer.value(), engine.state.epoch)
        logger.info('Train - Epoch: {} - Avg Time: {:3.2f}s'
              .format(engine.state.epoch, epoch_train_timer.value()))
        state = evaluator.run(val_loader)
        logger.info('Validate - Epoch: {} - Avg Time: {:3.2f}s'
              .format(engine.state.epoch, validate_timer.value()))
        writer.add_scalar("Validation/Loss",
                          state.metrics['running_val_loss'],
                          engine.state.epoch) # use trainer's state.epoch
        writer.add_scalar("Validation/Accuracy",
                          state.metrics['running_val_acc'],
                          engine.state.epoch)
        writer.add_scalar("Validation/Time", validate_timer.value(), engine.state.epoch)

    @evaluator.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_validation_terminal(engine):
        logger.info("Validate - Iter {}/{} - Avg accuracy: {:.3f} Avg loss: {:.3f}"
              .format(engine.state.iteration, len(val_loader),
                      engine.state.metrics['running_val_acc'], engine.state.metrics['running_val_loss']))

    @evaluator.on(Events.COMPLETED)
    def update_scheduler(engine):
        found_better = reduce_lr_scheduler.is_better(engine.state.metrics['running_val_loss'], reduce_lr_scheduler.best)
        reduce_lr_scheduler.step(engine.state.metrics['running_val_loss'])

        to_save = {
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': reduce_lr_scheduler.state_dict()
        }

        if found_better:
            torch.save(to_save, os.path.join(CKPT_DIR, 'BEST_weights.pt'))
            best_metrics = {
                'hparam/val_acc': evaluator.state.metrics['running_val_acc'],
                'hparam/val_loss': evaluator.state.metrics['running_val_loss'],
                'hparam/training_time': training_timer.value(),
            }
        filename = 'weights_val_loss_{:.3f}_val_acc_{:.3f}.pt'.format(engine.state.metrics['running_val_loss'],
                                                                      engine.state.metrics['running_val_acc'])
        torch.save(to_save, os.path.join(CKPT_DIR, filename))

    trainer.run(train_loader, max_epochs=2 if args.debug else config['max_epochs'])

    writer.add_hparams(config, best_metrics)

    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['tf', 's2s'])
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug-model', action='store_true', default=False)
    parser.add_argument('--log-root', type=str, default='./runs')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--multi-gpus', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    main(args)
