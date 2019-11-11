import os
import torch
import numpy as np
from model import Model
import editdistance as ed
import pdb

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

def load_checkpoint(config, all_data, ckpt_path):
    model = Model(config, all_data)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['start_learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, loss, epoch

def CER(outputs, targets):
    '''
    Calculate Character Error Rate (CER)
    :param outputs: [B, T, 1]
    :param targets: [B, T', 1]
    '''
    cer = []
    assert outputs.size(0) == targets.size(0)
    batch_size = outputs.size(0)
    for i in range(batch_size):
        cer.append(ed.eval(outputs[i], targets[i]))
    return cer

def WER(outputs, targets):
    '''
    Calculate Word Error Rate (WER)
    :param outputs: list of [T, 1]
    :param targets: list of [T', 1]
    '''

    assert len(outputs) == len(targets)
    batch_size = len(outputs)
    wer = [0 if outputs[i].size(0) == targets[i].size(0) and torch.eq(outputs[i], targets[i]) else 1 for i in range(batch_size)]
    return wer

def convert_to_text(tensor: torch.Tensor, dataset_all):
    '''
    :param tensor: [T, B, V]
    :return: list of [T, 1]
    '''
    batch_size, max_len, vocab_size = tensor.transpose(0, 1).size()
    results = []

    tensor, _ = tensor.max(dim=-1, keepdim=True) # [B, T, 1]
    eos = dataset_all.char2int[VNOnDBData.eos_char]
    eos = torch.tensor([[eos]]) # [1, 1]
    for i, item in enumerate(tensor):
        if torch.eq(item, eos):
            results.append(item[:i+1]) # including eos

    return results