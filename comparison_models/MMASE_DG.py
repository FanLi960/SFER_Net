"""Compact MMASE-DG-style multi-view domain-generalization network."""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        bottleneck = max(hidden_dim // 4, 4)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, bottleneck), nn.GELU(),
                                  nn.Linear(bottleneck, hidden_dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class MMASEDG(nn.Module):
    """Fuse spatial, spectral, and global views of the DE tensor."""

    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(62, hidden_dim), nn.GELU(),
                                     SqueezeExcitation(hidden_dim))
        self.spectral = nn.Sequential(nn.Linear(5, hidden_dim), nn.GELU(),
                                      SqueezeExcitation(hidden_dim))
        self.global_view = nn.Sequential(nn.Flatten(), nn.Linear(62 * 5, hidden_dim),
                                         nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
                                    nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        spatial = self.spatial(x.mean(dim=-1))
        spectral = self.spectral(x.mean(dim=1))
        global_view = self.global_view(x)
        embedding = self.fusion(torch.cat([spatial, spectral, global_view], dim=-1))
        return {"logits": self.classifier(embedding), "embedding": embedding}


if __name__ == "__main__":
    print(MMASEDG()(torch.randn(2, 62, 5))["logits"].shape)

