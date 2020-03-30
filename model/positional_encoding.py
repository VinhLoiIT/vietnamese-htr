import torch
import torch.nn as nn

class PositionalEncoding1d(nn.Module):

    def __init__(self, d_model, batch_first=False, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1) # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(1)

        self.batch_first = batch_first
        if self.batch_first:
            pe.transpose_(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding2d(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1) # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [B,C,H,W]
        '''
        x = x.permute(0,2,3,1) # [B,H,W,C]
        xshape = x.shape
        x = x.reshape(-1, x.size(2), x.size(3)) # [B*H,W,C]
        x = x + self.pe[:, :x.size(1), :]
        x = x.reshape(*xshape) # [B,H,W,C]
        x = x.permute(0,3,1,2) # [B,C,H,W]
        return self.dropout(x)
