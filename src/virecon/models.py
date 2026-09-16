from __future__ import annotations

import torch
from torch import nn


class MLPReconstructor(nn.Module):
    """Deterministic nonlinear baseline over the concatenated source montage."""

    def __init__(self, n_regions: int, feature_dim: int = 40, hidden: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_regions * feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        return self.network(sources)


class RegionalUncertaintyReconstructor(nn.Module):
    """A region-set encoder that predicts a nonlinear residual and uncertainty."""

    def __init__(self, n_regions: int, feature_dim: int = 40, hidden: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.region_embedding = nn.Parameter(torch.randn(n_regions, hidden) * 0.02)
        self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.mean = nn.Linear(hidden, feature_dim)
        self.log_variance = nn.Linear(hidden, feature_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, sources: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sources) + self.region_embedding.unsqueeze(0)
        scores = self.attention(encoded).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1)
        fused = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        latent = self.decoder(fused)
        return self.mean(latent), self.log_variance(latent).clamp(-8, 5)
