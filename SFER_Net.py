"""SFER-Net network definition.

Input
-----
DE features with shape ``[batch_size, 62, 5]``.

Output
------
A dictionary containing class logits, signed electrode-frequency evidence,
the hybrid adjacency matrix, and the global embedding.

This file contains the network only. Dataset loading, LOSO splitting, and the
episodic losses can be connected in the training code used by the researcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
]


def electrode_positions() -> np.ndarray:
    """Return approximate 2-D scalp coordinates in SEED channel order."""
    rows = [
        (["FP1", "FPZ", "FP2"], 1.00),
        (["AF3", "AF4"], 0.82),
        (["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"], 0.60),
        (["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"], 0.35),
        (["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"], 0.05),
        (["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"], -0.25),
        (["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"], -0.52),
        (["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"], -0.73),
        (["CB1", "O1", "OZ", "O2", "CB2"], -0.95),
    ]
    mapping = {}
    for names, y_coord in rows:
        x_coords = np.asarray([-0.38, 0.38]) if len(names) == 2 else np.linspace(-1.0, 1.0, len(names))
        for name, x_coord in zip(names, x_coords):
            mapping[name] = (float(x_coord), float(y_coord))
    return np.asarray([mapping[name] for name in CHANNEL_NAMES], dtype=np.float32)


def build_fixed_adjacency(neighborhood_size: int = 5) -> np.ndarray:
    """Build the normalized geometry-based electrode graph.

    The default reproduces the supplied experiment code: distance sorting
    includes the electrode itself, followed by its four closest neighbours.
    """
    positions = electrode_positions().astype(np.float64)
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    adjacency = np.zeros((62, 62), dtype=np.float64)
    for index in range(62):
        adjacency[index, np.argsort(distance[index])[:neighborhood_size]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T) + np.eye(62)
    degree = adjacency.sum(axis=1)
    adjacency /= np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1e-8))
    return adjacency.astype(np.float32)


@dataclass(frozen=True)
class SFERConfig:
    num_channels: int = 62
    num_bands: int = 5
    num_classes: int = 3
    hidden_dim: int = 64
    token_dim: int = 48
    graph_layers: int = 3
    residual_rank: int = 12
    dropout: float = 0.25
    fixed_graph_weight: float = 0.8
    residual_graph_weight: float = 0.2


class ResidualGraphBlock(nn.Module):
    """Aggregate graph messages and update every electrode representation."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(adjacency.unsqueeze(0), hidden)
        return self.norm(hidden + self.update(torch.cat([hidden, message], dim=-1)))


class SFERNet(nn.Module):
    """Evidence-guided hybrid graph network for EEG emotion recognition."""

    def __init__(self, config: SFERConfig | None = None):
        super().__init__()
        self.config = config or SFERConfig()
        cfg = self.config
        if (cfg.num_channels, cfg.num_bands) != (62, 5):
            raise ValueError("The released electrode graph expects 62 channels and 5 bands")

        # Project five DE bands into a channel-level hidden representation.
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.num_bands, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.channel_embedding = nn.Parameter(torch.empty(cfg.num_channels, cfg.hidden_dim))

        # Two low-rank factors parameterize the adaptive residual graph.
        self.adjacency_left = nn.Parameter(torch.empty(cfg.num_channels, cfg.residual_rank))
        self.adjacency_right = nn.Parameter(torch.empty(cfg.num_channels, cfg.residual_rank))
        self.graph_blocks = nn.ModuleList(
            [ResidualGraphBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.graph_layers)]
        )

        # Electrode-frequency token fusion and the shared evidence head.
        self.channel_token = nn.Linear(cfg.hidden_dim, cfg.token_dim)
        self.global_token = nn.Linear(cfg.hidden_dim, cfg.token_dim, bias=False)
        self.band_value = nn.Linear(1, cfg.token_dim, bias=False)
        self.band_embedding = nn.Parameter(torch.empty(cfg.num_bands, cfg.token_dim))
        self.token_norm = nn.LayerNorm(cfg.token_dim)
        self.evidence_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_dim, cfg.num_classes),
        )
        self.class_bias = nn.Parameter(torch.zeros(cfg.num_classes))
        fixed = torch.from_numpy(build_fixed_adjacency())
        self.register_buffer("fixed_adjacency", fixed, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.channel_embedding, std=0.02)
        nn.init.normal_(self.band_embedding, std=0.02)
        nn.init.normal_(self.adjacency_left, std=0.10)
        nn.init.normal_(self.adjacency_right, std=0.10)

    def adjacency_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the learned residual graph and the final hybrid graph."""
        cfg = self.config
        residual = F.relu(self.adjacency_left @ self.adjacency_right.T)
        residual = 0.5 * (residual + residual.T)
        residual = residual + torch.eye(62, device=residual.device, dtype=residual.dtype)
        residual = residual / residual.sum(dim=1, keepdim=True).clamp_min(1e-6)

        hybrid = cfg.fixed_graph_weight * self.fixed_adjacency
        hybrid = hybrid + cfg.residual_graph_weight * residual
        hybrid = hybrid / hybrid.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return residual, hybrid

    def forward(self, de_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.config
        expected = (cfg.num_channels, cfg.num_bands)
        if de_features.ndim != 3 or tuple(de_features.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, 62, 5], received {tuple(de_features.shape)}")

        hidden = self.input_projection(de_features)
        hidden = hidden + self.channel_embedding.unsqueeze(0)
        residual_adjacency, hybrid_adjacency = self.adjacency_components()
        for block in self.graph_blocks:
            hidden = block(hidden, hybrid_adjacency)

        global_context = hidden.mean(dim=1)
        tokens = self.channel_token(hidden).unsqueeze(2)
        tokens = tokens + self.global_token(global_context).unsqueeze(1).unsqueeze(2)
        tokens = tokens + self.band_value(de_features.unsqueeze(-1))
        tokens = tokens + self.band_embedding.unsqueeze(0).unsqueeze(0)
        tokens = self.token_norm(tokens)

        # E[b, c, i, f] is the signed evidence contributed by electrode i and
        # frequency band f to class c for sample b.
        evidence = self.evidence_head(tokens).permute(0, 3, 1, 2).contiguous()
        logits = evidence.sum(dim=(2, 3)) / math.sqrt(62 * 5) + self.class_bias
        return {
            "logits": logits,
            "evidence": evidence,
            "embedding": global_context,
            "hybrid_adjacency": hybrid_adjacency,
            "residual_adjacency": residual_adjacency,
        }


if __name__ == "__main__":
    model = SFERNet(SFERConfig(num_classes=3))
    example = torch.randn(4, 62, 5)
    output = model(example)
    print("logits:", output["logits"].shape)
    print("evidence:", output["evidence"].shape)
