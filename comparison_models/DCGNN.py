"""Compact DCGNN-style directed connectivity graph network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class DCGNN(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        self.source_factors = nn.Parameter(torch.randn(62, 12) * 0.05)
        self.target_factors = nn.Parameter(torch.randn(62, 12) * 0.05)
        self.input_layer = nn.Linear(5, hidden_dim)
        self.self_layer = nn.Linear(hidden_dim, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)

    def forward(self, x, grl_scale=1.0):
        directed = torch.sigmoid(self.source_factors @ self.target_factors.T)
        adjacency = directed + torch.eye(62, device=x.device, dtype=x.dtype)
        degree = adjacency.sum(-1).clamp_min(1e-6)
        adjacency = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
        hidden = F.gelu(self.input_layer(x))
        for _ in range(3):
            message = torch.einsum("ij,bjh->bih", adjacency, hidden)
            update = F.gelu(self.self_layer(hidden) + self.message_layer(message))
            hidden = self.norm(hidden + update)
        embedding = self.dropout(hidden.mean(dim=1))
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "domain_logits": self.domain_classifier(reversed_embedding),
                "adjacency": adjacency}


if __name__ == "__main__":
    print(DCGNN()(torch.randn(2, 62, 5))["logits"].shape)
