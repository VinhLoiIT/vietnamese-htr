import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_size, hidden_size, attn_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(feature_size, attn_size)
        self.Ua = nn.Linear(hidden_size, attn_size)
        self.va = nn.Linear(attn_size, 1)

        self.debug1 = None

    def forward(self, last_hidden, encoder_outputs):
        '''
        Input:
        :param last_hidden: [1, B, H]
        :param encoder_outputs: [num_pixels, B, C]
        Output:
        context: [1, B, C]
        weights: [num_pixels, B, 1]
        '''
        attn1 = self.Wa(encoder_outputs) # [num_pixels, B, A]
        attn2 = self.Ua(last_hidden) # [1, B, A]
        attn = self.va(torch.tanh(attn1 + attn2)) # [num_pixels, B, 1] - OK v1 is the same as v2
        
        weights = F.softmax(attn.squeeze(-1), 0).unsqueeze(-1) # [num_pixels, B, 1]
        
        self.debug1 = weights.transpose(0, -1)
        context = (weights * encoder_outputs).sum(0, keepdim=True) # [1, B, C]

        return context, weights

def run_v1(v1, last_hidden, image_features):
    contexts = []
    weights = []
    debug1 = []
    for t in range(len(last_hidden)):
        tgt = last_hidden[:t+1]
        print(t, '-'*30)
        print(tgt[[-1]])
        print(tgt)
        context, weight = v1.forward(tgt[[-1]], tgt)
        print(weight)
        print(context)
        print('-'*30)
        debug1.append(v1.debug1)
        # weight: [num_pixels, B, nhead, 1]
        # context: [1, B, nhead, A]
        contexts.append(context)
        weights.append(weight.transpose(0, 2))
    # debug1 = torch.cat(debug1)
    # context = torch.cat(contexts, dim=0)
    # weight = torch.cat(weights, dim=0)
    return contexts, weights, debug1

class Attentionv2(nn.Module):
    def __init__(self, feature_size, hidden_size, attn_size):
        super(Attentionv2, self).__init__()
        self.Wa = nn.Linear(feature_size, attn_size)
        self.Ua = nn.Linear(hidden_size, attn_size)
        self.va = nn.Linear(attn_size, 1)
        
        self.nhead = 1
        self.debug_sum = []
    
    def apply_mask(self, weights, attn_mask):
        saved_shape = weights.shape
        weights = weights.reshape(self.nhead, *attn_mask.shape)
        attn_mask = attn_mask.expand_as(weights)
        weights[~attn_mask] = float('-inf')
        weights = weights.reshape(*saved_shape)
        return weights
    
    def forward(self, last_hiddens, image_features, attn_mask=None):
        '''
        Input:
        - image_features: [S, B, A]
        - last_hiddens: [T, B, A]
        - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
        Output:
        - context: [T, B, A]
        - weights: [T, B, S, 1]
        '''
        b_image_features = self.Wa(image_features).transpose(0,1) # [B, S, A]
        b_last_hiddens = self.Ua(last_hiddens).transpose(0,1) # [B, T, A]
        weights = self.va(torch.tanh(b_image_features.unsqueeze(1) + b_last_hiddens.unsqueeze(2)))
        weights = weights.squeeze(-1) # [B, T, S]

        if attn_mask is not None:
            weights = self.apply_mask(weights, attn_mask)

        weights = F.softmax(weights, dim=-1) # [B,T,S]
        self.debug2 = weights.transpose(0, 1)

        print(weights)
        print(b_image_features)

        context = torch.matmul(weights, image_features.transpose(0,1)) # [B,T,S]x[B,S,A] = [B,T,A]
        print(context)
        # reshape
        context = context.transpose(0,1) # [T,B,A]
        weights = weights.transpose(0,1).unsqueeze(-1)

        return context, weights

if __name__ == "__main__":
    feature_size = 688
    hidden_size = 5
    attn_size = 256
    with torch.no_grad():
        v1 = Attention(hidden_size, hidden_size, hidden_size)
        v2 = Attentionv2(hidden_size, hidden_size, hidden_size)
        
        rand_w = torch.rand(v1.Wa.weight.data.shape)
        rand_b = torch.rand(v1.Wa.bias.data.shape)
        v1.Wa.weight.data = rand_w
        v2.Wa.weight.data = rand_w
        v1.Wa.bias.data = rand_b
        v2.Wa.bias.data = rand_b
                
        rand_w = torch.rand(v1.Ua.weight.data.shape)
        rand_b = torch.rand(v1.Ua.bias.data.shape)
        v1.Ua.weight.data = rand_w
        v2.Ua.weight.data = rand_w
        v1.Ua.bias.data = rand_b
        v2.Ua.bias.data = rand_b
                
        rand_w = torch.rand(v1.va.weight.data.shape)
        rand_b = torch.rand(v1.va.bias.data.shape)
        v1.va.weight.data = rand_w
        v2.va.weight.data = rand_w
        v1.va.bias.data = rand_b
        v2.va.bias.data = rand_b
        
        batch_size = 1
        hiddens_len = 3
        dummy_hiddens = torch.rand(hiddens_len, batch_size, hidden_size)
        
        v1_context, v1_weight, debug1 = run_v1(v1, dummy_hiddens, dummy_hiddens)
        
        attn_mask = torch.tril(torch.ones(batch_size, hiddens_len, hiddens_len)).bool()
        # attn_mask = None
        v2_context, v2_weight = v2.forward(dummy_hiddens, dummy_hiddens, attn_mask)
        debug2 = v2.debug2
        
        # print(v1_weight)
        # print(v2_weight.squeeze())
        # for context in v1_context:
        #     print(context)
        # print()
        # print(v2_context)