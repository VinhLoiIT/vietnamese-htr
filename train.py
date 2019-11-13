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
            inputs[len_.item():, i, :] = mask_value
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

all_data = VNOnDBData('./data/VNOnDB/train_word.csv')

image_transform = transforms.Compose([
    transforms.Grayscale(3),
    ScaleImageByHeight(32),
    transforms.ToTensor(),
])

train_data = VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', all_data, image_transform=image_transform)
validation_data = VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', all_data, image_transform=image_transform)
test_data = VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv', all_data, image_transform=image_transform)

train_loader = DataLoader(train_data, batch_size=31, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_loader = DataLoader(validation_data, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(n_epochs, model, optimizer, criterion, train_loader, max_length=MAX_LENGTH):

    losses = []
    cers = []

    t = tqdm.tqdm(train_loader)
    model.train()

    for epoch in range(args.epochs):
        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        for (inputs, targets, targets_one_hot, targets_lengths) in t:
            t.set_description(f'Epoch {epoch}/{n_epochs} (train={model.training})')

            inputs = inputs.to(device) # [B, C, H, W]
            targets = targets.float().to(device) # [T, B, 1]
            targets_one_hot = targets_one_hot.float().to(device) # [T, B, V]
            targets_lengths = targets_lengths # [B, 1]

            outputs, weights, decoded_lengths = model.forward(inputs, max_length, targets_one_hot, targets_lengths)
            # outputs: [T, B, V]
            # weights: [T, B, 1]

            # pdb.set_trace()
            outputs = mask_3d(outputs, decoded_lengths, 0)
            outputs = outputs.view(-1, all_data.vocab_size)
            targets = targets.view(-1).long()               

            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # Reset gradients
            optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            t.set_postfix(loss='{:05.3f}'.format(losses[-1]), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()

        save_checkpoint(model, optimizer, loss, epoch, './ckpt')

    return model, optimizer, losses
    # print(' End of training:  loss={:05.3f} , cer={:03.1f}'.format(np.mean(losses), np.mean(cers)*100))


def evaluate(model, val_loader):

    losses = []
    accs = []

    t = tqdm.tqdm(val_loader)
    model.eval()

    with torch.no_grad():
        for batch_image, targets, targets_one_hot, targets_lengths in t:
            t.set_description(' Evaluating... (train={})'.format(model.training))

            batch_image = batch_image.to(device)
            targets = targets.float().to(device)
            targets_one_hot = targets_one_hot.float().to(device)
            targets_lengths = targets_lengths.to(device)

            pdb.set_trace()
            outputs, weights = model(batch_image, targets_one_hot, targets_lengths)
            
            outputs = convert_to_text(outputs, all_data, device) # list of [T, 1] which T is variable length
            targets = convert_to_text(targets, all_data, device) # list of [T', 1] which T' is variable length

            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            wer = np.mean(WER(outputs, targets))
            losses.append(loss.item())
            wers.append(wer)
            t.set_postfix(avg_wer='{:05.3f}'.format(np.mean(wers)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig('data/att.png')
    print('  End of evaluation : loss {:05.3f} , acc {:03.1f}'.format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}



def run():
    global all_data
    #config_path = os.path.join('models', args.config)
    config = default_config

    #if not os.path.exists(config_path):
    #    raise FileNotFoundError

    #with open(config_path, 'r') as f:
    #    config = json.load(f)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # batch_size = config['batch_size']

    model = Model(4, 3, 96, 256, 256, device, all_data.vocab_size, 
        all_data.char2int[all_data.sos_char],
        all_data.char2int[all_data.pad_char],
        all_data.char2int[all_data.eos_char],
    )

    model = model.to(device)

    # Optimizer

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

    # model, optimizer, loss, _ = load_checkpoint(config, all_data, './ckpt/6.pt')
    # model = model.to(config['device'])
    # evaluate(model, val_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['start_learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    train(2, model, optimizer, criterion, train_loader, MAX_LENGTH)

    

    #     #evaluate(model, val_loader)

    #     # TODO implement save models function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    args, _ = parser.parse_known_args()
    run()
