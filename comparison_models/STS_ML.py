"""Compact STS-ML-style transformer and soft-mask network."""

import torch
import torch.nn as nn


class STSML(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.25):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by four")
        self.token_projection = nn.Linear(5, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.soft_mask = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                       nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        tokens = self.transformer(self.token_projection(x))
        attention = torch.sigmoid(self.soft_mask(tokens))
        embedding = (tokens * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1e-6)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "attention": attention.squeeze(-1)}


if __name__ == "__main__":
    print(STSML()(torch.randn(2, 62, 5))["logits"].shape)

