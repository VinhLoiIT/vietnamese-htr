import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        '''
        Shapes:
        -------
        logits: [N, num_classes]
        target: [N]
        '''
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1))
        prob = log_prob.detach().exp()

        if self.alpha is not None:
            alpha = self.alpha.to(target.device)
            alpha = alpha.expand(len(target), logits.size(-1)).detach().float()
            log_prob = (alpha * log_prob)
        
        loss = -(1 - prob)**self.gamma * log_prob

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            if self.alpha is None:
                return loss.mean()
            else:
                return loss.sum() / alpha.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('Invalid reduction')