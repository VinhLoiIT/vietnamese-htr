import os
import torch
from model import Model

def save_checkpoint(model, optimizer, loss, epoch, ckpt_dir):
    info = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    path = os.path.join(ckpt_dir, f'{epoch}.pt')
    print(f'Saving checkpoint to {path}')
    torch.save(info, path)

def load_checkpoint(ckpt_path):
    model = Model()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, loss, epoch