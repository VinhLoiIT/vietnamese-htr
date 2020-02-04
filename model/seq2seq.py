import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, cnn, decoder):
        super().__init__()
        self.cnn = cnn
        self.decoder = decoder

    def forward(self, image, target=None, output_weight=True):
        '''
        Input:
        :param image: [B, C, H, W]
        :param target: [L, B, V]
        Output:
        - predicts: [L, B, V]
        - weights: [L, B, H, W]
        '''
        batch_size, _, input_image_h, input_image_w = image.size()
        image_features = self.cnn(image) # [B, C', H', W']
        feature_image_h, feature_image_w = image_features.size()[-2:]
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.permute(2,0,1) # [S, B, C']

        predicts, weights = self.decoder.forward(image_features, target[1:], target[[0]])

        if not output_weight:
            return predicts, None

#         weights = weights.view(batch_size, -1, feature_image_h, feature_image_w)
#         weight_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((input_image_h, input_image_w)),
#         ])
#         weights_result = [weight_transform(weight) for weight in weights.cpu()]
        return predicts, weights

    def greedy(self, image, target=None):
        batch_size, _, input_image_h, input_image_w = image.size()
        image_features = self.cnn(image) # [B, C', H', W']
        feature_image_h, feature_image_w = image_features.size()[-2:]
        image_features = image_features.view(batch_size, self.cnn.n_features, -1) # [B, C', S=H'xW']
        image_features = image_features.permute(2,0,1) # [S, B, C']

        predicts, weights = self.decoder.greedy(image_features, target[[0]])
        
        return predicts, weights
