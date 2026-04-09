# LoRA.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=1.0, dropout=0.05):
        super().__init__()

        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = linear_layer.in_features
        out_dim = linear_layer.out_features

        # self.A = nn.Parameter(torch.empty(r, in_dim))
        # self.B = nn.Parameter(torch.randn(out_dim, r) * 0.01)
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)  # small
        self.B = nn.Parameter(torch.zeros(out_dim, r))  # start at zero
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

        # freeze base
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias

        original = F.linear(x, weight, bias)

        if not self.enabled:
            return original

        lora = self.dropout(x) @ self.A.T
        lora = lora @ self.B.T

        return original + self.scaling * lora

    def merge(self):
        delta_w = self.B @ self.A
        self.linear.weight.data += self.scaling * delta_w