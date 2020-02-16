import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import AdditiveAttention

# Old version (work only with for loops)
class Attention(nn.Module):
    def __init__(self, queries_size, keys_size, attn_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(keys_size, attn_size)
        self.Ua = nn.Linear(queries_size, attn_size)
        self.va = nn.Linear(attn_size, 1)

        self.debug1 = None

    def forward(self, keys, queries):
        '''
        Input:
        :param queries: [1, B, H]
        :param keys: [num_pixels, B, C]
        Output:
        context: [1, B, C]
        weights: [num_pixels, B, 1]
        '''
        attn1 = self.Wa(keys) # [num_pixels, B, A]
        attn2 = self.Ua(queries) # [1, B, A]

        attn = self.va(torch.tanh(attn1 + attn2)) # [num_pixels, B, 1] - OK v1 is the same as v2
        weights = F.softmax(attn.squeeze(-1), 0).unsqueeze(-1) # [num_pixels, B, 1]
        
        print(weights)
        context = (weights * keys).sum(0, keepdim=True) # [1, B, C]

        return context, weights

def run_v1(v1, queries, keys):
    contexts = []
    weights = torch.zeros(len(queries), queries.size(1), len(keys)) # [T,B,S]
    debug1 = []
    for t in range(len(queries)):
        # print(t, '-'*30)
        keys2 = keys[:t+1]
        context, weight = v1.forward(keys2, queries[[t]])
        # weight: S,B,1, S thay doi
        weights[[t], :, :t+1] = weight.transpose(0,-1)
        
        debug1.append(v1.debug1)
        # print('-'*30)
        # weight: [num_pixels, B, nhead, 1]
        # context: [1, B, nhead, A]
        contexts.append(context)
    context = torch.cat(contexts, dim=0)

    # print(weigh)
    return context, weights.transpose(0,1), debug1

if __name__ == "__main__":
    keys_size = 3
    queries_size = 2
    attn_size = 2
    with torch.no_grad():
        v1 = Attention(queries_size, keys_size, attn_size)
        v2 = AdditiveAttention(queries_size, keys_size, attn_size)        
        
        print(v1.Wa.weight.data.shape, v2.Wa.weight.data.shape)
        assert v1.Wa.weight.data.shape == v2.Wa.weight.data.shape
        assert v1.Wa.bias.data.shape == v2.Wa.bias.data.shape
        rand_w = torch.rand(v1.Wa.weight.data.shape).float()
        rand_b = torch.rand(v1.Wa.bias.data.shape).float()
        v1.Wa.weight.data = rand_w
        v2.Wa.weight.data = rand_w
        v1.Wa.bias.data = rand_b
        v2.Wa.bias.data = rand_b
                
        assert v1.Ua.weight.data.shape == v2.Ua.weight.data.shape
        assert v1.Ua.bias.data.shape == v2.Ua.bias.data.shape
        rand_w = torch.rand(v1.Ua.weight.data.shape).float()
        rand_b = torch.rand(v1.Ua.bias.data.shape).float()
        v1.Ua.weight.data = rand_w
        v2.Ua.weight.data = rand_w
        v1.Ua.bias.data = rand_b
        v2.Ua.bias.data = rand_b
                
        assert v1.va.weight.data.shape == v2.va.weight.data.shape
        assert v1.va.bias.data.shape == v2.va.bias.data.shape
        rand_w = torch.rand(v1.va.weight.data.shape).float()
        rand_b = torch.rand(v1.va.bias.data.shape).float()
        v1.va.weight.data = rand_w
        v2.va.weight.data = rand_w
        v1.va.bias.data = rand_b
        v2.va.bias.data = rand_b
        
        
        print('-'*50)
        print(v1.state_dict())
        print('-'*50)
        print(v2.state_dict())
        print('-'*50)
        
        batch_size = 1
        hiddens_len = 5
        keys_len = 3
        dummy_hiddens = torch.rand(hiddens_len, batch_size, queries_size, dtype=torch.float32)
        dummy_keys = torch.rand(keys_len, batch_size, keys_size, dtype=torch.float32)
        
        print('hidden',dummy_hiddens)
        print('keys', dummy_keys)

        v1_context, v1_weight, debug1 = run_v1(v1, dummy_hiddens, dummy_keys)
        # v2_context, v2_weight, _ = run_v2(v2, dummy_hiddens, dummy_hiddens)
        attn_mask = torch.tril(torch.ones(batch_size, hiddens_len, keys_len)).bool()
        print('Mask')
        print(attn_mask)
        v2_context, v2_weight = v2.forward(dummy_hiddens, dummy_keys, attn_mask, output_weights=True)

        print('-'*30)
        print('Weight')
        print(v1_weight)
        print(v2_weight)

        print('-'*30)
        print('Context')
        print(v1_context)
        print(v2_context)
        print(v1_context.isclose(v2_context))
        assert(v1_context.isclose(v2_context).all())
        
        # v1_context, v1_weight, debug1 = run_v1(v1, dummy_hiddens, dummy_hiddens)
        # # v2_context, v2_weight, _ = run_v2(v2, dummy_hiddens, dummy_hiddens)
        # attn_mask = torch.tril(torch.ones(batch_size, hiddens_len, hiddens_len)).bool()
        # v2_context, v2_weight = v2.forward(dummy_hiddens, dummy_hiddens, attn_mask, output_weights=True)
        
