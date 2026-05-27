import torch
import torch.nn as nn

from cs336_basics.Linear import Linear

def silu(x):
    return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)


    def forward(self, x):
        output = self.w2(silu(self.w1(x)) * self.w3(x))

        return output