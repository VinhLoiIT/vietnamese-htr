# import argparse
import os
import numpy as np
import ignite
import ignite.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import collate_fn, get_dataset, vocab_size
from model.decoder import Decoder
from model.encoder import Encoder
from utils import ScaleImageByHeight

CKPT_DIR = './ckpt'
if not os.path.exists(CKPT_DIR):
    os.mkdir(CKPT_DIR)

config = {
    'batch_size': 32,
    'hidden_size': 256,
    'attn_size': 256,
    'max_length': 10,
    'n_epochs_decrease_lr': 15,
    'start_learning_rate': 1e-5,  # NOTE: paper start with 1e-8
    'end_learning_rate': 1e-11,
    'depth': 4,
    'n_blocks': 3,
    'growth_rate': 96,
    'max_epochs': 100,
    'weight_decay': 0,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    encoder = Encoder(config['depth'], config['n_blocks'], config['growth_rate'])
    decoder = Decoder(encoder.n_features, config['hidden_size'], vocab_size, config['attn_size'])

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.RMSprop(params, lr=config['start_learning_rate'], weight_decay=config['weight_decay'])
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=config['n_epochs_decrease_lr'],
        threshold=0.005,
        threshold_mode='abs',
        min_lr=config['end_learning_rate'],
        verbose=True)

    image_transform = transforms.Compose([
        transforms.Grayscale(3),
        ScaleImageByHeight(128),
        transforms.ToTensor(),
    ])

    train_data = get_dataset('train', image_transform)
    validation_data = get_dataset('val', image_transform)

    # NOTE: try on small subset of data to make sure it works before running on all data!
    # train_loader = DataLoader(train_data, batch_size=config['batch_size'], 
    #                           shuffle=False, collate_fn=collate_fn, num_workers=12, sampler=torch.utils.data.SubsetRandomSampler(np.random.permutation(256)))
    # val_loader = DataLoader(validation_data, batch_size=config['batch_size'], 
    #                         shuffle=False, collate_fn=collate_fn, num_workers=12, sampler=torch.utils.data.SubsetRandomSampler(np.random.permutation(256)))

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, num_workers=12)

    val_loader = DataLoader(validation_data, batch_size=config['batch_size'], 
                            shuffle=False, collate_fn=collate_fn, num_workers=12)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    writer = SummaryWriter()

    def step_train(engine, batch):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        imgs, targets, targets_onehot, lengths = batch

        imgs = imgs.to(device)
        targets = targets.to(device)
        targets_onehot = targets_onehot.to(device)

        img_features = encoder(imgs)
        outputs, _ = decoder(img_features, targets_onehot[1:], targets_onehot[[0]])

        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, (lengths - 1).squeeze())[0]
        packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
            targets[1:].squeeze(), (lengths - 1).squeeze())[0]

        loss = criterion(packed_outputs, packed_targets)
        loss.backward()
        optimizer.step()

        return packed_outputs, packed_targets

    def step_val(engine, batch):
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            imgs, targets, targets_onehot, lengths = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_onehot = targets_onehot.to(device)

            img_features = encoder(imgs)
            outputs, _ = decoder(img_features, targets_onehot[1:], targets_onehot[[0]])

            packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
                outputs, (lengths - 1).squeeze())[0]
            packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
                targets[1:].squeeze(), (lengths - 1).squeeze())[0]

            return packed_outputs, packed_targets

    trainer = ignite.engine.Engine(step_train)
    evaluator = ignite.engine.Engine(step_val)

    metrics.RunningAverage(metrics.Loss(criterion)).attach(trainer, 'running_train_loss')
    metrics.RunningAverage(metrics.Accuracy()).attach(trainer, 'running_train_acc')
    metrics.RunningAverage(metrics.Loss(criterion)).attach(evaluator, 'running_val_loss')
    metrics.RunningAverage(metrics.Accuracy()).attach(evaluator, 'running_val_acc')

    training_timer = ignite.handlers.Timer(average=True).attach(trainer)

    @trainer.on(ignite.engine.Events.STARTED)
    def start_training(engine):
        print('Config..')
        print('='*60)
        print(config)
        print('='*60)
        print('Start training..')

    @trainer.on(ignite.engine.Events.COMPLETED)
    def end_training(engine):
        print('Training completed!!')
        print('Total training time: {:3.2f}s'.format(training_timer.value()))

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log_training_tensorboard(engine):
        writer.add_scalar("Train/Loss", engine.state.metrics['running_train_loss'], engine.state.iteration)
        writer.add_scalar("Train/Accuracy", engine.state.metrics['running_train_acc'], engine.state.iteration)

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=50))
    def log_training_terminal(engine):
        print("Train - Epoch: {} - Iter {} - Accuracy: {:.3f} Loss: {:.3f}"
              .format(engine.state.epoch, engine.state.iteration, 
                      engine.state.metrics['running_train_acc'], engine.state.metrics['running_train_loss']))

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def validate(engine):
        evaluator.run(val_loader)

    @evaluator.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_validation_tensorboard(engine):
        writer.add_scalar("Validation/Loss", engine.state.metrics['running_val_loss'], engine.state.epoch)
        writer.add_scalar("Validation/Accuracy", engine.state.metrics['running_val_acc'], engine.state.epoch)

    @evaluator.on(ignite.engine.Events.ITERATION_COMPLETED(every=50))
    def log_validation_terminal(engine):
        print("Validate - Epoch: {} - Iter {}/{} - Avg accuracy: {:.3f} Avg loss: {:.3f}"
              .format(engine.state.epoch, engine.state.iteration, len(val_loader),
                      engine.state.metrics['running_val_acc'], engine.state.metrics['running_val_loss']))

    @evaluator.on(ignite.engine.Events.COMPLETED)
    def update_scheduler(engine):
        found_better = reduce_lr_scheduler.is_better(engine.state.metrics['running_val_loss'], reduce_lr_scheduler.best)
        reduce_lr_scheduler.step(engine.state.metrics['running_val_loss'])

        to_save = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 
                   'optimizer': optimizer.state_dict(), 'lr_scheduler': reduce_lr_scheduler.state_dict()}
        if found_better:
            torch.save(to_save, os.path.join(CKPT_DIR, 'BEST_weights.pt'))
        torch.save(to_save, os.path.join(CKPT_DIR, 'weights.pt'))

    trainer.run(train_loader, max_epochs=config['max_epochs'])
    # trainer.run(train_loader, max_epochs=2)

    writer.close()

main()
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--resume', type=str)
#     parser.add_argument('--config', type=str, default=os.path.join(CFG_DIR, 'config.json'))
#     args = parser.parse_args()

#     run(args)
