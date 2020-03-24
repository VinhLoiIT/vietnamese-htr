import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class CustomCNN(nn.Module):
    '''
    An Efficient End-to-End Neural Model for Handwritten Text Recognition (2018)
    '''
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, 3, padding=1)),
            ('lrelu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('bn1', nn.BatchNorm2d(16)),

            ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
            ('lrelu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool2d(2)),
            ('bn2', nn.BatchNorm2d(32)),

            ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
            ('lrelu3', nn.LeakyReLU()),
            ('bn3', nn.BatchNorm2d(64)),

            ('conv4', nn.Conv2d(64, 64, 3, padding=1)),
            ('lrelu4', nn.LeakyReLU()),
            ('bn4', nn.BatchNorm2d(64)),

            ('conv5', nn.Conv2d(64, 128, 3, padding=1)),
            ('lrelu5', nn.LeakyReLU()),
            ('pool5', nn.MaxPool2d((1,2))),
            ('bn5', nn.BatchNorm2d(128)),

            ('conv6', nn.Conv2d(128, 128, 3, padding=1)),
            ('lrelu6', nn.LeakyReLU()),
            ('pool6', nn.MaxPool2d((1,2))),
            ('bn6', nn.BatchNorm2d(128)),

            ('conv7', nn.Conv2d(128, 128, (2,2))),
            ('lrelu7', nn.LeakyReLU()),
            ('bn7', nn.BatchNorm2d(128)),
        ]))
    
    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":
    with torch.no_grad():
        x = torch.rand(1,1,32,128)
        cnn = CustomCNN()
        cnn.eval()

        output = cnn(x)
        print(output.shape)