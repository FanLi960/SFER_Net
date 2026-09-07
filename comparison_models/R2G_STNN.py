"""Compact R2G-STNN-style regional-to-global recurrent network."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SFER_Net import electrode_positions


class R2GSTNN(nn.Module):
    """Aggregate 62 electrodes into nine regions before recurrent fusion."""

    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.25):
        super().__init__()
        positions = electrode_positions()
        bins = np.digitize(positions[:, 1], [-0.80, -0.60, -0.38, -0.10,
                                                  0.20, 0.48, 0.72, 0.91])
        region = np.zeros((9, 62), dtype=np.float32)
        for index in range(9):
            selected = bins == index
            region[index, selected] = 1.0 / max(int(selected.sum()), 1)
        self.register_buffer("region", torch.from_numpy(region))
        self.region_projection = nn.Linear(9, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                                bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x):
        # [B, 62, 5] -> five band steps, each represented by nine regions.
        sequence = torch.einsum("rc,bcf->bfr", self.region, x)
        hidden, _ = self.recurrent(F.gelu(self.region_projection(sequence)))
        weights = torch.softmax(self.attention(hidden), dim=1)
        embedding = (weights * hidden).sum(dim=1)
        return {"logits": self.classifier(embedding), "embedding": embedding,
                "attention": weights.squeeze(-1)}


if __name__ == "__main__":
    print(R2GSTNN()(torch.randn(2, 62, 5))["logits"].shape)

