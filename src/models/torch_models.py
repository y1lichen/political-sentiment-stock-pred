from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.direction = nn.Linear(hidden_dim, 1)
        self.ret = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.direction(h).squeeze(-1), self.ret(h).squeeze(-1)


class EventGatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.direction = nn.Linear(hidden_dim, 1)
        self.ret = nn.Linear(hidden_dim, 1)
        self.trade_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return (
            self.direction(h).squeeze(-1),
            self.ret(h).squeeze(-1),
            self.trade_gate(h).squeeze(-1),
        )

