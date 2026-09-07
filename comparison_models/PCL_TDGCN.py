"""Compact PCL-TDGCN-style bandwise dynamic graph and prototype network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLTDGCN(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.band_node = nn.Linear(1, hidden_dim)
        self.self_layer = nn.Linear(hidden_dim, hidden_dim)
        self.message_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        band_embeddings = []
        adjacency_mean = 0.0
        for band in range(5):
            values = x[:, :, band:band + 1]
            similarity = torch.einsum("bci,bdi->bcd", values, values)
            identity = torch.eye(62, device=x.device, dtype=x.dtype).unsqueeze(0)
            adjacency = torch.sigmoid(similarity) + identity
            degree = adjacency.sum(-1).clamp_min(1e-6)
            adjacency = adjacency / torch.sqrt(degree[:, :, None] * degree[:, None, :])
            nodes = F.gelu(self.band_node(values))
            message = torch.einsum("bij,bjh->bih", adjacency, nodes)
            nodes = self.norm(nodes + F.gelu(self.self_layer(nodes) + self.message_layer(message)))
            band_embeddings.append(nodes.mean(dim=1))
            adjacency_mean = adjacency_mean + adjacency.mean(dim=0)

        sequence = torch.stack(band_embeddings, dim=1)
        hidden, _ = self.recurrent(sequence)
        embedding = self.dropout(hidden[:, -1])
        prototypes = F.normalize(self.prototypes, dim=-1)
        contrastive_logits = F.normalize(embedding, dim=-1) @ prototypes.T / 0.15
        logits = self.classifier(embedding) + contrastive_logits
        return {"logits": logits, "embedding": embedding, "prototypes": prototypes,
                "adjacency": adjacency_mean / 5.0}


if __name__ == "__main__":
    print(PCLTDGCN()(torch.randn(2, 62, 5))["logits"].shape)
