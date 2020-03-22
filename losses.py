import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        '''
        Shapes:
        -------
        logits: [N, num_classes]
        target: [N]
        '''
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.detach().exp()
        loss = torch.pow(1 - prob.gather(-1, target.unsqueeze(-1)), self.gamma) * log_prob.gather(-1, target.unsqueeze(-1))
        loss = -loss.mean()
        return loss
