import torch.nn as nn
import torchvision
import torch.nn.functional as F

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
    
'''
LeNet-5
https://engmrk.com/lenet-5-a-classic-cnn-architecture/
https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
https://www.kaggle.com/usingtc/lenet-with-pytorch
'''
class LeNetFE(FE):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.activation = F.relu()

        self.n_features = 84

    def forward(self, inputs):
        x = self.activation(self.conv1(inputs))
        x = self.max_pool_1(x)
        x = self.activation(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_n_features(self):
        return self.n_features