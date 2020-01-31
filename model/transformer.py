import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder


class TransformerModel(nn.Module):
    def __init__(self, cnn_features, vocab_size, d_model):
        super(TransformerModel, self).__init__()
        decoder_layer = nn.modules.TransformerDecoderLayer(d_model, 8)
        self.transformer = nn.modules.TransformerDecoder(decoder_layer, num_layers=6, norm=None)
        
        self.linear_img = nn.Linear(cnn_features, d_model)
        self.linear_text = nn.Linear(vocab_size, d_model)
        
        self.linear_output = nn.Linear(d_model, vocab_size)
        
        pass
    
    def forward(self, img_features, target_onehots, targets, targets_length, PAD_CHAR_int):
        img_features = self.linear_img(img_features)
        # Optional: PositionEncoding for imgs
        # here ...
        
        targets_onehot = self.linear_text(target_onehots.float())
        # Optional: PositionEncoding for text
        # here ...
        
        tgt_mask = generate_square_subsequent_mask(torch.max(targets_length)).to(targets.device) # ignore SOS_CHAR
        tgt_key_padding_mask = (targets == PAD_CHAR_int).squeeze().transpose(0,1).to(targets.device) # [B,T]
        
        memory_mask = None
        memory_key_padding_mask = None
        outputs = self.transformer.forward(targets_onehot, img_features,
                                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask, 
                                           memory_key_padding_mask=memory_key_padding_mask)
        # [T, N, d_model]
        outputs = self.linear_output(outputs) # [T, N, vocab_size]
        return outputs

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
