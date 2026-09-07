"""Compact DGCNN-style network with a trainable global adjacency matrix."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency):
    degree = adjacency.sum(-1).clamp_min(1e-6)
    return adjacency / torch.sqrt(degree[:, None] * degree[None, :])


class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, adjacency):
        messages = torch.einsum("ij,bjf->bif", adjacency, features)
        return F.gelu(self.linear(messages))


class DGCNN(nn.Module):
    """Learn electrode relations jointly with the emotion classifier."""

    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.node_factors = nn.Parameter(torch.randn(62, 16) * 0.05)
        self.layers = nn.ModuleList([
            GraphLayer(5, hidden_dim),
            GraphLayer(hidden_dim, hidden_dim),
            GraphLayer(hidden_dim, hidden_dim),
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        identity = torch.eye(62, device=x.device, dtype=x.dtype)
        adjacency = F.softplus(self.node_factors @ self.node_factors.T) + identity
        adjacency = normalize_adjacency(adjacency)
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        embedding = self.dropout(hidden.mean(dim=1))
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "adjacency": adjacency}


if __name__ == "__main__":
    print(DGCNN()(torch.randn(2, 62, 5))["logits"].shape)

