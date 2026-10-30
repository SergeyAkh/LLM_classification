import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=8.0, dropout=0.05):
        super().__init__()

        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = linear_layer.in_features
        out_dim = linear_layer.out_features

        # Freeze pretrained weights.
        for p in self.linear.parameters():
            p.requires_grad = False

        # Trainable LoRA matrices.
        self.A = nn.Parameter(
            torch.randn(r, in_dim) * 0.01
        )
        self.B = nn.Parameter(
            torch.zeros(out_dim, r)
        )

        self.dropout = nn.Dropout(dropout)
        self.enabled = True

        # DoRA: trainable magnitude for each output row.
        # Initialize from the pretrained weight norms.
        with torch.no_grad():
            weight = self.linear.weight.detach().float()
            magnitude = torch.linalg.vector_norm(
                weight, dim=1, keepdim=True
            )
        self.magnitude = nn.Parameter(magnitude.squeeze(1))

    def get_weight(self):
        # Compute the effective weight:
        # W' = m * (W + scaling * B @ A) / ||W + scaling * B @ A||
        # where normalization is performed per output row.

        weight = self.linear.weight
        weight_fp = weight.float()

        if self.enabled:
            delta = self.B @ self.A
            adapted_weight = (
                weight_fp + self.scaling * delta.float()
            )
        else:
            adapted_weight = weight_fp

        # Row-wise L2 norm.
        norm = torch.linalg.vector_norm(
            adapted_weight, dim=1, keepdim=True
        ).clamp_min(1e-8)

        direction = adapted_weight / norm

        # Scale each output row by its trainable magnitude.
        effective_weight = (
            self.magnitude.float().unsqueeze(1) * direction
        )

        return effective_weight.to(weight.dtype)

    def forward(self, x):
        weight = self.get_weight()
        bias = self.linear.bias
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def merge(self):
        # Merge the current effective DoRA weight into the base.
        self.linear.weight.copy_(self.get_weight())
        # Prevent accidental double application.
        self.enabled = False