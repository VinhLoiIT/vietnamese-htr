import torch.nn as nn
import torchvision

class FE(nn.Module):
    def __init__(self):
        super().__init__()

    def get_cnn(self):
        raise NotImplementedError()
    
    def get_n_features(self):
        raise NotImplementedError()
    
    def forward(self, inputs):
        '''
        :param inputs: [B, C, H, W]
        :returms: [B, C', H', W']
        '''
        return self.get_cnn()(inputs) # [B, C', H', W']

class DenseNetFE(FE):
    def __init__(self, depth, n_blocks, growth_rate):
        super().__init__()
        densenet = torchvision.models.DenseNet(
            growth_rate=growth_rate,
            block_config=[depth]*n_blocks
        )

        self.cnn = densenet.features
        self.n_features = densenet.classifier.in_features

    def get_cnn(self):
        return self.cnn
    
    def get_n_features(self):
        return self.n_features