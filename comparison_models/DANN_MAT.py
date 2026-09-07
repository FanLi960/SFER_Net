"""Compact DANN-MAT-style multi-adversarial network."""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class DANNMAT(nn.Module):
    """Use several subject discriminators with a shared EEG encoder."""

    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(62 * 5, hidden_dim * 2), nn.GELU(),
            nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, num_domains)) for _ in range(3)
        ])

    def forward(self, x, grl_scale=1.0):
        embedding = self.encoder(x)
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "domain_logits": [head(reversed_embedding) for head in self.domain_heads]}


if __name__ == "__main__":
    print(DANNMAT()(torch.randn(2, 62, 5))["logits"].shape)

