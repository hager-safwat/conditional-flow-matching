"""Velocity network v_theta(x, t, c)."""

import math

import torch
from torch import nn


class SinusoidalTime(nn.Module):
    """Sinusoidal embedding of continuous time t in [0, 1].

    t is multiplied by 100 before the sinusoids so that the frequencies,
    designed for integer positions, still vary over the unit interval.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: shape (B,), float in [0, 1] -> (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        args = t[:, None] * freqs[None, :] * 100.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityField(nn.Module):
    """MLP predicting a 2-D velocity from the point x, the time t and the
    class label c (looked up in a small learned embedding table)."""

    def __init__(self, n_classes=2, data_dim=2, t_dim=64, c_dim=32, hidden=256):
        super().__init__()
        self.time_embed = SinusoidalTime(t_dim)
        self.class_embed = nn.Embedding(n_classes, c_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim + c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t, c):
        h = torch.cat([x, self.time_embed(t), self.class_embed(c)], dim=-1)
        return self.net(h)
