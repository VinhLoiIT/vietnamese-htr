import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, depth, n_blocks, growth_rate):
        super(Encoder, self).__init__()

        self.cnn = torchvision.models.DenseNet(
            growth_rate=growth_rate,
            block_config=[depth]*n_blocks
        ).features

        # TODO: fix me
        self.n_features = self.cnn.norm5.num_features
    
    def forward(self, inputs):
        '''
        :param inputs: [B, C, H, W]
        :returms: [num_pixels, B, C']
        '''
        batch_size = inputs.size(0)
        outputs = self.cnn(inputs) # [B, C', H', W']
        outputs = outputs.view(batch_size, self.n_features, -1) # [B, C', H' x W'] == [B, C', num_pixels]
        outputs = outputs.permute(2, 0, 1) # [num_pixels, B, C']
        return outputs