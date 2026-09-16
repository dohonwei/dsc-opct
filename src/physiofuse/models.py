from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * grad, None


class TemporalEncoder(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.normalize = nn.LayerNorm(features)
        self.rnn = nn.GRU(features, hidden, batch_first=True, bidirectional=True)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.rnn(self.normalize(x))
        weights = torch.softmax(self.score(states).squeeze(-1), dim=1)
        return self.dropout((states * weights.unsqueeze(-1)).sum(1))


class PhysioFuseDG(nn.Module):
    def __init__(self, mode: str, hidden: int = 64, dropout: float = 0.2, domains: int = 2) -> None:
        super().__init__()
        self.mode = mode
        self.eeg_encoder, self.pps_encoder = TemporalEncoder(40, hidden, dropout), TemporalEncoder(6, hidden, dropout)
        dim = hidden * 2
        if mode == "early":
            self.early_encoder = TemporalEncoder(46, hidden, dropout)
        elif mode == "physiofuse":
            self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.classifier = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.domain = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, domains))

    def forward(self, eeg: torch.Tensor, pps: torch.Tensor, reverse_weight: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        he, hp = self.eeg_encoder(eeg), self.pps_encoder(pps)
        if self.mode == "eeg":
            z = he
        elif self.mode == "pps":
            z = hp
        elif self.mode == "early":
            z = self.early_encoder(torch.cat([eeg, pps], dim=-1))
        else:
            weights = torch.softmax(self.gate(torch.cat([he, hp], dim=-1)), dim=-1)
            z = weights[:, :1] * he + weights[:, 1:] * hp
        return self.classifier(z), self.domain(ReverseGradient.apply(z, reverse_weight)), he, hp
