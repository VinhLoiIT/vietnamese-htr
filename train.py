import torch
import numpy as np
import editdistance
import matplotlib.pyplot as plt
import tqdm
import argparse
import json
import os
import pdb
from PIL import Image
from model import Model
from dataset import VNOnDB, VNOnDBData, collate_fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from utils import CER, WER, save_checkpoint, load_checkpoint, convert_to_text, ScaleImageByHeight, LabelToInt

def mask_3d(inputs, inputs_lens, mask_value=0.):
    '''
    :param inputs: [T, B, V]
    :param inputs_lens: [B, 1]
    '''
    max_len = max(inputs_lens)
    for i, len_ in enumerate(inputs_lens):
        if len_ < max_len.item():
            inputs[len_.item():, i] = torch.tensor([mask_value])
    return inputs


default_config = {
#   'batch_size': 31,
  'hidden_size': 256,
  'attn_size': 256,
  'max_length': 10,
  'n_epochs_decrease_lr': 15,
  'start_learning_rate': 0.00000001,
  'end_learning_rate': 0.00000000001,
  'depth': 4,
  'n_blocks': 3,
  'growth_rate': 96,
}
MAX_LENGTH = 10
CKPT_DIR = './ckpt'

if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)

all_data = VNOnDBData('./data/VNOnDB/all_word.csv') #replace train_word by all_word

image_transform = transforms.Compose([
    transforms.Grayscale(3),
    ScaleImageByHeight(32),
    transforms.ToTensor(),
])

train_data = VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', all_data, image_transform=image_transform)
validation_data = VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', all_data, image_transform=image_transform)
test_data = VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv', all_data, image_transform=image_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_loader = DataLoader(validation_data, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)

# train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(validation_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(epoch, model, optimizer, criterion, max_length=MAX_LENGTH):

    model.train()
    wers = [] 
    losses = []
    t = tqdm.tqdm(train_loader)

    for i, (inputs, targets, targets_one_hot, targets_lengths) in enumerate(t):
        t.set_description(f'Epoch {epoch} (train={model.training})')

        inputs = inputs.to(device) # [B, C, H, W]
        targets = targets.to(device) # [T, B]
        targets_one_hot = targets_one_hot.to(device) # [T, B, V]
        targets_lengths = targets_lengths # [B, 1]

        optimizer.zero_grad()

        outputs, weights, decoded_lengths = model(inputs, max_length, targets_one_hot, targets_lengths)
        # outputs: [T, B, V]
        # weights: [T, B, 1]
        wers += words_correct_list(outputs, targets, targets_lengths)

        outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, decoded_lengths.squeeze())[0]
        targets = torch.nn.utils.rnn.pack_padded_sequence(targets, targets_lengths.squeeze())[0]
        loss = criterion(outputs, targets)

        losses += [loss.item()] * inputs.size(0)
        # Reset gradients
        # Compute gradients
        loss.backward()
        optimizer.step()

        t.set_postfix(avg_loss=f'{np.mean(losses):05.3f}', acc=f'{np.mean(wers):05.3f}')
        t.update()

    return model, optimizer, np.mean(losses), np.mean(wers)


def validate(model, criterion, max_length=MAX_LENGTH):

    losses = []
    wers = []

    t = tqdm.tqdm(val_loader)
    model.eval()

    with torch.no_grad():
        for (inputs, targets, targets_one_hot, targets_lengths) in t:
            t.set_description(f'Eval (train={model.training})')
            inputs = inputs.to(device) # [B, C, H, W]
            targets = targets.to(device) # [T, B]
            targets_one_hot = targets_one_hot.to(device) # [T, B, V]
            targets_lengths = targets_lengths # [B, 1]

            outputs, weights, decoded_lengths = model.forward(inputs, max_length, targets_one_hot, targets_lengths)
            # outputs: [T, B, V]
            # weights: [T, B, 1]
            wers += words_correct_list(outputs, targets, targets_lengths)

            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, decoded_lengths.squeeze())[0]
            targets = torch.nn.utils.rnn.pack_padded_sequence(targets, targets_lengths.squeeze())[0]

            loss = criterion(outputs, targets)

            losses += [loss.item()] * inputs.size(0)
            

            t.set_postfix(avg_loss=f'{np.mean(losses):05.3f}', acc=f'{np.mean(wers):05.3f}')
            t.update()

    return model, np.mean(losses), np.mean(wers)


def train(model, info, max_length=MAX_LENGTH):
    
    if info is None:
        info = dict()
        info['train_losses'] = []
        info['train_accs'] = []
        info['val_losses'] = []
        info['val_accs'] = []
        info['model_state_dict'] = None
        info['epoch'] = 0
        info['lr'] = 1e-3
        info['optimizer_state_dict'] = None
        info['count_not_improve'] = 0
        info['best_val_accs'] = 0

    if info['model_state_dict'] is not None:
        model.load_state_dict(info['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'])
    if info['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(info['optimizer_state_dict'])

    model.train()
    criterion = torch.nn.NLLLoss().to(device)

    print('Start training...')
    # pdb.set_trace()
    while True:
        info['epoch'] += 1
        model, optimizer, train_loss, train_acc = train_one_epoch(info['epoch'], model, optimizer, criterion, max_length)
        model, val_loss, val_acc = validate(model, criterion)

        info['train_losses'].append(train_loss)
        info['train_accs'].append(train_acc)
        info['val_losses'].append(val_loss)
        info['val_accs'].append(val_acc)
        info['optimizer_state_dict'] = optimizer.state_dict()
        info['model_state_dict'] = model.state_dict()

        if val_acc >= info['best_val_accs']:
            info['count_not_improve'] = 0
            info['best_val_accs'] = val_acc

            ckpt_path = os.path.join(CKPT_DIR, f'BEST.pt')
            torch.save(info, ckpt_path)
            print(f'Saved at epoch = {info["epoch"]}')
        else:
            info['count_not_improve'] += 1 

        if info['count_not_improve'] >= 15:
            info['count_not_improve'] = 0
            info['lr'] /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'])

        if info['lr'] <= 1e-5:
            break

    return model, optimizer, losses

def words_correct_list(predict, target, targets_lengths):
    '''
    :param predict: [T, B, V]
    :param target: [T, B]
    :param targets_lengths: [B, 1]
    '''
    predict = predict.argmax(-1).transpose(0, 1).long() # [B, T]
    target = target.transpose(0, 1).long() # [B, T]
    batch_size = target.size(0)

    result = [1 if predict[i, :targets_lengths[i]].equal(target[i, :targets_lengths[i]]) else 0 for i in range(batch_size)]
    return result

def run():
    global all_data
    config = default_config

    model = Model(4, 3, 96, 256, 256, device, all_data.vocab_size, 
        all_data.char2int[all_data.sos_char],
        all_data.char2int[all_data.pad_char],
        all_data.char2int[all_data.eos_char],
    )

    model = model.to(device)

    print('=' * 60)
    print(model)
    print('=' * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(' (' + k + ') : ' + str(v))
    print()
    print('=' * 60)

    # print('\nInitializing weights...')
    # for name, param in model.named_parameters():
    #     if 'bias' in name:
    #         torch.nn.init.constant_(param, 0.0)
    #     elif 'weight' in name:
    #         torch.nn.init.xavier_normal_(param)

    info = None
    if args.resume is not None:
        try:
            info = torch.load(args.resume)
        except FileNotFoundError as e:
            print(e)

    train(model, info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--resume', type=str)
    args, _ = parser.parse_known_args()
    run()
