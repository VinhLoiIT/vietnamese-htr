import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, cnn, vocab_size, hidden_size, attn_size, decoder_attn='additive'):
        super().__init__()
        self.cnn = cnn
        self.decoder = Decoder(decoder_attn, self.cnn.get_n_features(), hidden_size, vocab_size, attn_size)

    def forward(self, image, target):
        '''
        Input:
        :param image: [B, C, H, W]
        :param target: [B, L, V]
        Output:
        - predicts: [B, L, V]
        - weights: [B, L, H, W]
        '''
        batch_size, _, input_image_h, input_image_w = image.size()
        image_features = self.cnn(image) # [B, C', H', W']
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.transpose(1,2) # [B, S, C']

        predicts = self.decoder(image_features, target)
        return predicts

    def greedy(self, image, start_input, max_length=10, output_weights=False):
        batch_size, _, input_image_h, input_image_w = image.size()
        image_features = self.cnn(image) # [B, C', H', W']
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.transpose(1,2) # [B,S,C']

        predicts, weights = self.decoder.greedy(image_features, start_input, max_length=max_length)

        if output_weights:
            # TODO: scale weights to fit image size and return
            return predicts, weights
        else:
            return predicts, None
