"""Compact BiDANN-style bilateral domain-adversarial network."""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from SFER_Net import electrode_positions


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class BiDANN(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        x_position = electrode_positions()[:, 0]
        self.register_buffer("left", torch.as_tensor(np.where(x_position <= 0)[0]))
        self.register_buffer("right", torch.as_tensor(np.where(x_position >= 0)[0]))
        self.left_encoder = nn.Sequential(nn.Flatten(), nn.LazyLinear(hidden_dim),
                                          nn.GELU(), nn.Dropout(dropout))
        self.right_encoder = nn.Sequential(nn.Flatten(), nn.LazyLinear(hidden_dim),
                                           nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                    nn.GELU(), nn.LayerNorm(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                               nn.GELU(), nn.Linear(hidden_dim, num_domains))

    def forward(self, x, grl_scale=1.0):
        left = self.left_encoder(x.index_select(1, self.left))
        right = self.right_encoder(x.index_select(1, self.right))
        embedding = self.fusion(torch.cat([left, right], dim=-1))
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "domain_logits": self.domain_classifier(reversed_embedding)}


if __name__ == "__main__":
    print(BiDANN()(torch.randn(2, 62, 5))["logits"].shape)
