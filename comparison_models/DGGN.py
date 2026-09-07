"""Compact DGGN-style sample-dependent graph generation network."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from SFER_Net import build_fixed_adjacency


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class DGGN(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        self.register_buffer("fixed", torch.from_numpy(build_fixed_adjacency()))
        self.graph_generator = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 8))
        self.input_layer = nn.Linear(5, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)

    def forward(self, x, grl_scale=1.0):
        factors = F.normalize(self.graph_generator(x), dim=-1)
        adjacency = torch.sigmoid(torch.einsum("bik,bjk->bij", factors, factors) /
                                  math.sqrt(factors.shape[-1]))
        identity = torch.eye(62, device=x.device, dtype=x.dtype).unsqueeze(0)
        adjacency = adjacency * self.fixed.unsqueeze(0) + identity
        degree = adjacency.sum(-1).clamp_min(1e-6)
        adjacency = adjacency / torch.sqrt(degree[:, :, None] * degree[:, None, :])
        hidden = F.gelu(self.input_layer(x))
        for _ in range(2):
            message = torch.einsum("bij,bjh->bih", adjacency, hidden)
            hidden = self.norm(hidden + F.gelu(self.message_layer(message)))
        embedding = self.dropout(hidden.mean(dim=1))
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "domain_logits": self.domain_classifier(reversed_embedding),
                "adjacency": adjacency}


if __name__ == "__main__":
    print(DGGN()(torch.randn(2, 62, 5))["logits"].shape)

