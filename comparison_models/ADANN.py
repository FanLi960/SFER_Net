"""Compact ADANN-style adversarial prototype network."""

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


class ADANN(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(62 * 5, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.1)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)

    def forward(self, x, grl_scale=1.0):
        embedding = F.normalize(self.encoder(x), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 30.0)
        logits = scale * embedding @ prototypes.T
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": logits, "embedding": embedding,
                "domain_logits": self.domain_classifier(reversed_embedding),
                "prototypes": prototypes}


if __name__ == "__main__":
    print(ADANN()(torch.randn(2, 62, 5))["logits"].shape)

