import torch
import numpy as np
import tqdm
import argparse
import os
import pdb
from model.encoder import Encoder
from model.decoder import Decoder
from dataset import get_dataset, collate_fn, vocab_size
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from utils import ScaleImageByHeight, AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter

config = {
    'batch_size': 64,
    'hidden_size': 256,
    'attn_size': 256,
    'max_length': 10,
    'n_epochs_decrease_lr': 15,
    'start_learning_rate': 1e-5,  # NOTE: paper start with 1e-8
    'end_learning_rate': 1e-11,
    'depth': 4,
    'n_blocks': 3,
    'growth_rate': 96,
}

MAX_LENGTH = config['max_length']
CKPT_DIR = './ckpt'

if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)

image_transform = transforms.Compose([
    transforms.Grayscale(3),
    ScaleImageByHeight(32),
    transforms.ToTensor(),
])

train_data = get_dataset('train', image_transform)
validation_data = get_dataset('val', image_transform)
test_data = get_dataset('test', image_transform)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=8)
val_loader = DataLoader(validation_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=8)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)

# train_loader = DataLoader(train_data, batch_size=64,
#                           shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(validation_data, batch_size=32,
#                         shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_data, batch_size=32,
#                          shuffle=False, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(info, train_loader, encoder, decoder, optimizer, criterion, writer, log_interval=100):
    encoder.train()
    decoder.train()

    losses = AverageMeter()
    accs = AverageMeter()
    
    print('Training')

    for i, (imgs, targets, targets_onehot, lengths) in enumerate(train_loader):

        optimizer.zero_grad()

        imgs = imgs.to(device)
        targets = targets.to(device)
        targets_onehot = targets_onehot.to(device)

        img_features = encoder(imgs)
        outputs, weights = decoder(img_features, targets_onehot)

        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, lengths.squeeze())[0]
        packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
            targets.squeeze(), lengths.squeeze())[0]

        loss = criterion(packed_outputs, packed_targets)
        acc = accuracy(packed_outputs, packed_targets)

        total_characters = lengths.sum().item()
        losses.update(loss, total_characters)
        accs.update(acc, total_characters)

        loss.backward()
        optimizer.step()

        info['train_step'] += 1
        writer.add_scalar('Train/Loss', loss.item(), info['train_step'])
        writer.add_scalar('Train/Accuracy', acc, info['train_step'])

        if (i+1) % log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(info['epoch'], i, len(train_loader),
                                                                    loss=losses,
                                                                    accs=accs))
    return losses.avg, accs.avg


def validate(info, val_loader, encoder, decoder, criterion, writer, log_interval=100):
    losses = AverageMeter()
    accs = AverageMeter()

    encoder.eval()
    decoder.eval()
    
    print('Validating')
    with torch.no_grad():
        for i, (imgs, targets, targets_onehot, lengths) in enumerate(val_loader):

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            img_features = encoder(imgs)
            outputs, weights = decoder(img_features, targets_onehot)

            packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
                outputs, lengths.squeeze())[0]
            packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
                targets.squeeze(), lengths.squeeze())[0]
            loss = criterion(packed_outputs, packed_targets)
            acc = accuracy(packed_outputs, packed_targets)

            total_characters = lengths.sum().item()
            losses.update(loss, total_characters)
            accs.update(acc, total_characters)

            info['val_step'] += 1
            writer.add_scalar('Validate/Loss', loss.item(), info['val_step'])
            writer.add_scalar('Validate/Accuracy', acc, info['val_step'])

            if (i+1) % log_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(info['epoch'], i, len(val_loader),
                                                                        loss=losses,
                                                                        accs=accs))
    return losses.avg, accs.avg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(info, is_best=False):
    filename = 'weights.pt'
    ckpt_path = os.path.join(CKPT_DIR, filename)
    torch.save(info, ckpt_path)
    if is_best:
        ckpt_path = os.path.join(CKPT_DIR, 'BEST_' + filename)
        torch.save(info, ckpt_path)

def train(info: dict):
    if info is None:
        info = dict()
        info['encoder_state'] = None
        info['decoder_state'] = None
        info['optimizer_state'] = None
        info['scheduler_state'] = None
        info['train_step'] = 0
        info['val_step'] = 0
        info['epoch'] = 0
        info['best_val_accs'] = 0
        info['lr'] = config['start_learning_rate']
    else:
        print('Resume training...')
        print(f'Epoch: {info["epoch"]}')
        print(f'Best validation accuracy: {info["best_val_accs"]:05.3f}')
        print(f'Learning rate: {info["lr"]}')

    encoder = Encoder(
        config['depth'], config['n_blocks'], config['growth_rate'])
    if info['encoder_state'] is not None:
        encoder.load_state_dict(info['encoder_state'])

    decoder = Decoder(encoder.n_features,
                      config['hidden_size'], vocab_size, config['attn_size'])
    if info['decoder_state'] is not None:
        decoder.load_state_dict(info['decoder_state'])

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=config['start_learning_rate'])
    if info['optimizer_state'] is not None:
        optimizer.load_state_dict(info['optimizer_state'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=config['n_epochs_decrease_lr'],
        min_lr=config['end_learning_rate'], verbose=True)
    if info['scheduler_state'] is not None:
        scheduler.load_state_dict(info['scheduler_state'])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter()

    print('Start training...')
    while True:
        info['epoch'] += 1
        train_loss, train_acc = train_one_epoch(
            info, train_loader, encoder, decoder, optimizer, criterion, writer, log_interval=100)
        val_loss, val_acc = validate(
            info, val_loader, encoder, decoder, criterion, writer, log_interval=100)
        scheduler.step(val_acc)

        info['optimizer_state'] = optimizer.state_dict()
        info['scheduler_state'] = scheduler.state_dict()
        info['encoder_state'] = encoder.state_dict()
        info['decoder_state'] = decoder.state_dict()
        info['lr'] = get_lr(optimizer)

        if val_acc > info['best_val_accs']:
            info['best_val_accs'] = val_acc
            save_checkpoint(info, True)
        else:
            save_checkpoint(info, False)

        if info['lr'] <= config['end_learning_rate']:
            print('Reach min learning rate. Stop training...')
            break

    return encoder, decoder, info


def run(args):
    
    print('=' * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(' (' + k + ') : ' + str(v))
    print()
    print('=' * 60)

    info = None
    if args.resume is not None:
        try:
            info = torch.load(args.resume)
        except FileNotFoundError as e:
            print(e)
    
    encoder, decoder, info = train(info)
    print('=' * 60)
    print('Number training epochs: ', info['epoch'])
    print('Best validation accuracy: ', info['best_val_accs'])
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    run(args)
