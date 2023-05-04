import torch
import torch.nn as nn


# ------ utils ------ #


class TCM(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_cats,
                 eps=2e-5,
                 act=nn.LeakyReLU()
               ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_cats = num_cats
        self.eps = eps
        self.mid_act = act
        
        self.layer = nn.Linear(self.dim_in, self.dim_out)
        self.weight = torch.nn.Parameter(
            torch.Tensor(num_cats, dim_out))
        self.weight.data.fill_(1.0)
        self.bias = torch.nn.Parameter(torch.Tensor(
            num_cats, dim_out))
        self.bias.data.zero_()
        
    def normalize(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        mask = (std == 0).squeeze()
        output = (input - mean) / (std + self.eps)
        output[mask] = 0
        return output

    def forward(self, input, cats):
        h = self.layer(input)
        out = self.normalize(h)
        shape = [h.size(0), self.dim_out
                    ] + (h.dim() - 2) * [1]
        weight = self.weight.index_select(0, cats).view(shape)
        bias = self.bias.index_select(0, cats).view(shape)
        out = out * weight + bias
        out = self.mid_act(out)
        return out
