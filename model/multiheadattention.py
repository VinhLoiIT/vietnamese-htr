import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, attn_size, nhead=1):
        super().__init__()
        if attn_size % nhead != 0:
            raise ValueError(f'attn_size should divisible nhead: {attn_size} % {nhead} != 0')

        logging.debug('Create MultiHeadAttention with feature_size={}, hidden_size={}, attn_size={}, nhead={}'.format(
                feature_size, hidden_size, attn_size, nhead))

        if feature_size != attn_size or hidden_size != attn_size:
            logging.debug('Convert dimmension = True')
            self.Wc = nn.Linear(feature_size, attn_size)
            self.Uc = nn.Linear(hidden_size, attn_size)
            self._convert_dim = True
        else:
            logging.debug('Convert dimmension = False')
            self._convert_dim = False
        self.nhead = nhead
        self.head_attn_size = attn_size // nhead
        logging.debug('Head attention size = {}'.format(self.head_attn_size))

    def attn(self, image_features, last_hiddens, attn_mask):
        logging.error('This is an abstract method that subclass should override')
        raise NotImplementedError()

    def multihead_view(self, tensor):
        result =  tensor.view(tensor.shape[0], tensor.shape[1]*self.nhead, self.head_attn_size)
        logging.debug('multihead view of {} is {}'.format(tensor.shape, result.shape))
        return result

    def apply_mask(self, weights, attn_mask):
        saved_shape = weights.shape
        weights = weights.reshape(self.nhead, *attn_mask.shape)
        attn_mask = attn_mask.expand_as(weights)
        weights[~attn_mask] = float('-inf')
        weights = weights.reshape(*saved_shape)
        return weights

    def forward(self, image_features, last_hiddens, attn_mask=None):
        '''
        Input:
        :param last_hiddens: [T, B, H]
        :param image_features: [S, B, C]
        Output:
        context: [T, B, A]
        weights: [T, B, S, 1]
        '''
        batch_size = image_features.size(1)
        image_features_len = image_features.size(0)
        if self._convert_dim:
            image_features = self.Wc(image_features) # [S, B, A]
            last_hiddens = self.Uc(last_hiddens) # [T, B, A]

        image_features = self.multihead_view(image_features)
        last_hiddens = self.multihead_view(last_hiddens)

        context, weight = self.attn(image_features, last_hiddens, attn_mask)
        # weight: [L, B*nhead, S, 1]
        # context: [1, B*nhead, A]

        context = context.view(-1, batch_size, self.nhead, self.head_attn_size)
        weight = weight.view(-1, batch_size, self.nhead, image_features_len, 1)

        weight = torch.mean(weight, dim=-3, keepdim=False) # weight: [num_pixels, B, 1]
        context = context.reshape(context.shape[0], context.shape[1], -1) # context: [1, B, nhead x head_attn_size] = [1, B, A]
        return context, weight


class AdditiveAttention(MultiHeadAttention):
    def __init__(self, feature_size, hidden_size, attn_size, nhead=1):
        super().__init__(feature_size, hidden_size, attn_size, nhead)
        self.Wa = nn.Linear(self.head_attn_size, self.head_attn_size)
        self.Ua = nn.Linear(self.head_attn_size, self.head_attn_size)
        self.v = nn.Linear(self.head_attn_size, 1)

    def attn(self, image_features, last_hiddens, attn_mask=None):
        '''
        Input:
        - image_features: [S, B, A]
        - last_hiddens: [T, B, A]
        - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
        Output:
        - context: [T, B, A]
        - weights: [T, B, S, 1]
        '''
        logger = logging.getLogger('AdditiveAttention')
        logger.debug('image_features.shape = {}'.format(image_features.shape))
        b_image_features = self.Wa(image_features).transpose(0,1) # [B, S, A]
        logger.debug('b_image_features.shape = {}'.format(b_image_features.shape))
        b_last_hiddens = self.Ua(last_hiddens).transpose(0,1) # [B, T, A]
        logger.debug('b_last_hiddens.shape = {}'.format(b_last_hiddens.shape))
        #b_image_features2 = b_image_features.unsqueeze(1) # [B, 1, S, A]
        #b_last_hiddens2 = b_last_hiddens.unsqueeze(2) # [B, T, 1, A]
        #weights = self.v(torch.tanh(b_image_features2 + b_last_hiddens2)) # [B, T, S, 1]
        weights = self.v(torch.tanh(b_image_features.unsqueeze(1) + b_last_hiddens.unsqueeze(2)))
        logger.debug('weights.shape = {}'.format(weights.shape))
        weights = weights.squeeze(-1) # [B, T, S]
        logger.debug('weights.shape = {}'.format(weights.shape))

        if attn_mask is not None:
            weights = self.apply_mask(weights, attn_mask)

        weights = F.softmax(weights, dim=-1) # [B,T,S]
        logger.debug('weights.shape = {}'.format(weights.shape))

        context = weights.bmm(image_features.transpose(0,1)) # [B,T,S]x[B,S,A] = [B,T,A]
        logger.debug('context.shape = {}'.format(context.shape))
        # reshape
        context = context.transpose(0,1) # [T,B,A]
        weights = weights.transpose(0,1).unsqueeze(-1)

        logger.debug('context.shape = {}, weights.shape = {}'.format(context.shape, weights.shape))

        return context, weights

class ScaleDotProductAttention(MultiHeadAttention):
    def __init__(self, feature_size, hidden_size, attn_size, nhead=1):
        super().__init__(feature_size, hidden_size, attn_size, nhead)

    def attn(self, image_features, last_hiddens, attn_mask=None):
        '''
        Input:
        - image_features: [S, B, A] - Keys, Values
        - last_hiddens: [T, B, A] - Queryes
        - attn_mask: [B, T, S]
        Output:
        - contexts: [T, B, A]
        - weights: [T, B, S, 1]
        '''
        attn_dim = image_features.size(-1)
        b_image_features = image_features.transpose(0, 1) # [B, S, A]
        b_last_hiddens = last_hiddens.transpose(0, 1) # [B, T, A]

        # [B,T,A] x [B,A,S] = [B,T,S]
        matmul = b_last_hiddens.bmm(b_image_features.transpose(1, 2))
        scaled = matmul / attn_dim # [B,T,S]
        if attn_mask is not None:
            weights = self.apply_mask(scaled, attn_mask)
        weights = F.softmax(scaled, dim=-1) # [B,T,S]

        # [B,T,S] x [B,S,A] = [B,T,A]
        context = weights.bmm(b_image_features)

        # form shape
        context = context.transpose(0, 1) # [B,T,A] -> [T,B,A]
        weights = weights.unsqueeze(-1).transpose(0, 1) # [B,T,S] -> [B,T,S,1] -> [T,B,S,1]

        return context, weights


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with torch.no_grad():
        feature_size = 688
        hidden_size = 150
        attn_size = 256
        batch_size = 10
        image_features_len = 100
        last_hiddens_len = 20
        dummy_image_features = torch.rand(image_features_len, batch_size, feature_size)
        dummy_last_hiddens = torch.rand(last_hiddens_len, batch_size, hidden_size)
        logging.info('dummy_image_features.shape = {}, dummy_last_hiddens.shape = {}'.format(dummy_image_features.shape, dummy_last_hiddens.shape))

        # test using mask
        scale_dot_product = ScaleDotProductAttention(hidden_size, hidden_size, hidden_size, nhead=10)
        additive = AdditiveAttention(hidden_size, hidden_size, hidden_size, nhead=10)
        attn_mask = torch.tril(torch.ones(batch_size,last_hiddens_len, last_hiddens_len)).bool()
        output, weight = additive(dummy_last_hiddens, dummy_last_hiddens, attn_mask)
        logging.info('output additive not use masking = {}'.format(output.shape))
        output, weight = scale_dot_product(dummy_last_hiddens, dummy_last_hiddens, attn_mask)
        logging.info('output scale_dot not use masking = {}'.format(output.shape))
        
        # test not using mask
        scale_dot_product = ScaleDotProductAttention(feature_size, hidden_size, attn_size, nhead=8)
        additive = AdditiveAttention(feature_size, hidden_size, attn_size, nhead=8)
        output, weight = additive(dummy_image_features, dummy_last_hiddens)
        logging.info('output additive not use masking = {}'.format(output.shape))
        output, weight = scale_dot_product(dummy_image_features, dummy_last_hiddens)
        logging.info('output scale_dot not use masking = {}'.format(output.shape))
    logging.info('Done')