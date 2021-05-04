import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, encoder_dim, att_type):
        super(Attention, self).__init__()
        if att_type == "additive":
            self.attention_layer = AdditiveAttention(encoder_dim)
        elif att_type == "dotprod":
            self.attention_layer = DotProdAttention(encoder_dim)
        else:
            raise ValueError("Attention mechanism not defined: " + str(att_type))

    def forward(self, encoder_output, hidden_state, sum_context=False):
        context, alpha = self.attention_layer(encoder_output, hidden_state)
        if sum_context:
            context = context.sum(1)
        return context, alpha


# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(AdditiveAttention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, encoder_output, hidden_state):
        # encoder_output ------ torch.Size([24, 49, 2048])
        # hidden_state ------ torch.Size([24, 512])
        U_h = self.U(hidden_state).unsqueeze(1)
        # Uh ------ torch.Size([24, 1, 512])
        W_s = self.W(encoder_output)
        # Ws (s=encoder_output) ------ torch.Size([24, 49, 512])
        att = self.tanh(W_s + U_h)
        # a = tanh(Ws + Uh) ------ torch.Size([24, 49, 512])
        e = self.v(att).squeeze(2)
        # e = V^tanh(Ws + Uh) ------ torch.Size([24, 49])
        alpha = self.softmax(e)
        # alpha ------ torch.Size([24, 49])
        context = encoder_output * alpha.unsqueeze(2)
        # context ------ torch.Size([24, 49, 2048])
        return context, alpha


# Dot-Product Based or General Multiplicative Attention
class DotProdAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(DotProdAttention, self).__init__()
        self.scale = 1.0/np.sqrt(encoder_dim)
        self.softmax = nn.Softmax(dim=2)
        # raise NotImplementedError("TODO: Implement attention layer")

    def forward(self, encoder_output, hidden_state):
        # Verify sizes
        # query: [B,Q] (hidden state, decoder output, etc.)
        # keys: [T,B,K] (encoder outputs)
        # values: [T,B,V] (encoder outputs)
        # assume Q == K
        
        # compute energy
        query = hidden_state
        keys = encoder_output
        values = encoder_output
        
        ## Source: https://gist.github.com/shreydesai/3b4c5ee9ea135a7693c5886078257371

        # print("Hidden_state shape: {}".format(hidden_state.shape))
        # print(encoder_output.shape)
        # print(encoder_output)
        query = query.unsqueeze(1) # [B,Q] -> [B,1,Q]
        keys = keys.permute((0,2,1)) # [T,B,K] -> [B,K,T]
        # print(query.shape)
        # print(keys.shape)
        energy = torch.bmm(query, keys) # [B,1,Q]*[B,K,T] = [B,1,T]
        # energy = self.softmax(energy.mul_(self.scale))
        
        # # apply mask, renormalize
        # energy = energy*mask
        # energy.div(energy.sum(2, keepdim=True))

        # weight values
        # values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
        # values = values.permute(0, )
        combo = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]

        # return context, alpha
        return combo, energy


if __name__ == "__main__":
    model = Attention(512, "dotprod").to(device)
    model.eval()
    print(model)
    encoder_output = torch.randn(2, 256, 512).to(device)
    v_embedding = torch.randn(2, 512).to(device)
    with torch.no_grad():
        output, alpha = model.forward(encoder_output, v_embedding)
    print(output.size())
    print(alpha.size())
