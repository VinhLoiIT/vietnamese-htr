import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    '''
    Spatial Transformer Network
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.pool = nn.AdaptiveAvgPool2d((3,3))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self._init_weights()
        
    def _init_weights(self):
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, images):
        '''
        Shapes:
        -------
            - images: [B,C,H,W]
        '''
        x = self.localization(images) # [B,C',H',W']
        x = self.pool(x) # [B,C',3,3]
        x = x.reshape(-1, 10 * 3 * 3)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, images.size(), align_corners=True)
        images = F.grid_sample(images, grid, align_corners=True)

        return images