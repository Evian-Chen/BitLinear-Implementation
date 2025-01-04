"""
File: BitLinear.py
Name: Evian(Yan-he) Chen
----------------------------------------------------
This file implements BitLinear based on
the original paper: https://arxiv.org/pdf/2310.11453
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear:
    def __init__(self, in_features, out_features, bits=8, epsilon=1e-5, linear=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.epsilon = epsilon
        self.linear = linear

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.layer_norm = nn.LayerNorm(normalized_shape=in_features, eps=epsilon)
        
        # for testing
        self.original_weight = self.weight

    def __call__(self, x):
        # weight quantization
        alpha = self.mean_alpha(self.weight)
        self.weight = self.Sign(self.weight - alpha)

        # LayerNorm
        x = self.layer_norm(x)

        # activations quantization
        Qb = 2 ** (self.bits-1)
        gama = self.absmax(x)
        if self.linear: 
            x = torch.clamp(x*(Qb/gama), min=-Qb+self.epsilon, max=Qb-self.epsilon)
        else:  # non-linear layer, e.g., ReLU
            eta = torch.min(x)
            x = torch.clamp((x-eta)*(Qb/gama), min=self.epsilon, max=Qb-self.epsilon)

        # linear projection
        out = F.linear(x, self.weight)

        # dequantization
        beta = self.l1_norm(self.weight)
        out = (out*gama*beta) / Qb

        return out

    def Sign(self, weight):
        return torch.where(weight>0, 1.0, -1.0)

    def mean_alpha(self, weight):
        n, m = weight.shape
        return weight / (n*m)

    def absmax(self, x):
        return torch.max(torch.abs(x))
    
    def l1_norm(self, weight):
        n, m = weight.shape
        norm = torch.sum(torch.abs(weight))
        return norm / (n*m)


# ----- test ----- #

bit_linear = BitLinear(in_features=4, out_features=2)
x = torch.randn(3, 4)  # [batch_size=3, in_features=4]
output = bit_linear(x)

print("Input:\n", x)
print("Output:\n", output)
print("Binary weights:\n", bit_linear.weight)
print("Original weigth:\n", bit_linear.original_weight)
         