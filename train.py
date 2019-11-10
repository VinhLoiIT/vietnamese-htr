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
from dataset import VNOnDB, VNOnDBData, to_batch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms

def mask_3d(inputs, seq_len, mask_value=0.):
    assert inputs.size(0) == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, 'The size of inputs must be 2 or 3, received {}'.format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


def train(model, optimizer, train_loader, state):
    epoch, n_epochs = state

    losses = []
    cers = []

    criterion = CrossEntropyLoss()
    t = tqdm.tqdm(train_loader)
    model.train()


    mask_value = 0
    # if self.loss_type == 'NLL': # ie softmax already on outputs
    #     mask_value = -float('inf')
    #     print(torch.sum(logits, dim=2))
    # else:
    #     mask_value = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for (batch_image, targets, targets_one_hot, targets_lengths) in t:
        t.set_description('Epoch {:.0f}/{:.0f} (train={})'.format(epoch, n_epochs, model.training))

        batch_image = batch_image.to(device)
        targets = targets.to(device)
        targets_one_hot = targets_one_hot.to(device)
        targets_lengths = targets_lengths.to(device)

        outputs, weights = model.forward(batch_image, targets_one_hot, targets_lengths)
        # outputs: [T, B, V]
        # weights: [T, B, 1]

        outputs = mask_3d(outputs.transpose(1, 0), targets_lengths, mask_value)
        outputs = outputs.contiguous().view(-1, len(all_data.alphabets))

        #pdb.set_trace()
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    return model, optimizer, loss
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
            targets = targets.to(device)
            targets_one_hot = targets_one_hot.to(device)
            targets_lengths = targets_lengths.to(device)

            outputs, weights = model(batch_image, targets_one_hot, targets_lengths)
            
            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            acc = 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)
            losses.append(loss.item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig('data/att.png')
    print('  End of evaluation : loss {:05.3f} , acc {:03.1f}'.format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}

default_config = {
  'batch_size': 31,
  'hidden_size': 256,
  'attn_size': 256,
  'n_epochs_decrease_lr': 15,
  'start_learning_rate': 0.00000001,
  'end_learning_rate': 0.00000000001,
  'device': 'cuda',
  'depth': 4,
  'n_blocks': 3,
  'growth_rate': 96,
}

class ScaleByHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, image):
        width, height = image.size
        factor = self.target_height / height
        new_width = int(width * factor)
        new_height = int(height * factor)
        image = image.resize((new_width, new_height))
        return image

class LabelToOneHot(object):
    def __init__(self, all_data):
        self.all_data = all_data

    def __call__(self, label):
        label = list(label) + [VNOnDBData.eos_char]
        label = [self.all_data.char2int[character] for character in label]
        return self.all_data.int2onehot(label)

def run():
    global all_data
    #config_path = os.path.join('models', args.config)
    config = default_config

    #if not os.path.exists(config_path):
    #    raise FileNotFoundError

    #with open(config_path, 'r') as f:
    #    config = json.load(f)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['batch_size']

    all_data = VNOnDBData('./data/VNOnDB/train_word.csv')

    image_transform = transforms.Compose([
        #transforms.Resize((320, 480)),
        ScaleByHeight(32),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    label_transform = transforms.Compose([
        LabelToOneHot(all_data),
        transforms.ToTensor(),
    ])

    train_data = VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv', image_transform=image_transform, label_transform=label_transform)
    validation_data = VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv', image_transform=image_transform, label_transform=label_transform)
    test_data = VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=to_batch, num_workers=10)
    val_loader = DataLoader(validation_data, batch_size=1, shuffle=False, collate_fn=to_batch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=to_batch)

    # Models
    model = Model(config, all_data)

    model = model.to(config['device'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['start_learning_rate'])

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

    for epoch in range(args.epochs):
        run_state = (epoch, args.epochs)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer, loss = train(model, optimizer, train_loader, run_state)
        save_checkpoint(model, optimizer, loss, epoch, './ckpt')

        #evaluate(model, val_loader)

        # TODO implement save models function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    args, _ = parser.parse_known_args()
    run()




