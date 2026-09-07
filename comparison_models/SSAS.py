"""Compact SSAS-style source-selection and adversarial network."""

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


class SSAS(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, num_domains=15, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(62 * 5, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)
        self.source_selector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                             nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, grl_scale=1.0):
        embedding = self.encoder(x)
        sample_weight = 0.5 + torch.sigmoid(self.source_selector(embedding)).squeeze(-1)
        reversed_embedding = GradientReverse.apply(embedding, grl_scale)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "domain_logits": self.domain_classifier(reversed_embedding),
                "sample_weight": sample_weight}


if __name__ == "__main__":
    print(SSAS()(torch.randn(2, 62, 5))["logits"].shape)

