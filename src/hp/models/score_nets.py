"""Score networks with the time in every layer (docs/experiments/2026-11-14_score_network_architectures.md, section 3).

TimeResNet is the network of RSGM's net="timeres": a residual MLP in which every layer gets its own projection of a
shared embedding of the noise level, with a normalization in every layer, as DPoser's TimeFC (and as PoseRFM's tMLP
conditions every layer on t), on the features RSGM gives it (the rotation matrices, optionally their forward
kinematics and a condition):

    temb = SiLU(Linear(sinusoidal(c)))                      shared, c = log(s(t)) / 4 the noise level
    h    = SiLU(Norm(Linear(features) + Linear_t(temb)))
    B blocks:  h <- h + [Dropout(SiLU(Norm(Linear(h) + Linear_t(temb))))] twice
    out  = Linear(h)                                        initialized to zero: the score starts at 0

The noise level is log s (the Brownian time of the IGSO(3) noise) rather than t: small times, where the score changes
fast and pose completion works, are spread out (t = 0.002 and t = 0.02 differ by 2.3 in log s, as t = 0.1 and t = 1
by 2.6). The sinusoidal embedding is DPoser's (positional, of 1000 c, frequencies 1 ... 1/10000).
"""
import math

import torch
from torch import nn


def timestep_embedding(levels: torch.Tensor, dim: int, max_positions: int = 10_000) -> torch.Tensor:
    """The sinusoidal embedding [B, dim] of real levels [B], as DPoser's (and DDPM's) get_timestep_embedding."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=levels.device) * -(math.log(max_positions) / (half - 1)))
    angles = levels.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return nn.functional.pad(emb, (0, 1)) if dim % 2 else emb


class _Layer(nn.Module):
    """Linear(h) + Linear_t(temb), then Norm, SiLU and Dropout."""

    def __init__(self, width_in: int, width: int, embed: int, dropout: float):
        super().__init__()
        self.dense, self.dense_t = nn.Linear(width_in, width), nn.Linear(embed, width)
        self.norm, self.act, self.dropout = nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.dense(h) + self.dense_t(temb))))


class TimeResNet(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int = 1024, blocks: int = 2, embed: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = embed
        self.time = nn.Sequential(nn.Linear(embed, embed), nn.SiLU())
        self.first = _Layer(dim_in, width, embed, dropout)
        self.blocks = nn.ModuleList(nn.ModuleList([_Layer(width, width, embed, dropout) for _ in range(2)])
                                    for _ in range(blocks))
        self.last = nn.Linear(width, dim_out)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, features: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """features [B, dim_in], level [B] (the noise level c) -> [B, dim_out]."""
        temb = self.time(timestep_embedding(1000 * level, self.embed).to(features.dtype))
        h = self.first(features, temb)
        for block in self.blocks:
            h1 = h
            for layer in block:
                h1 = layer(h1, temb)
            h = h + h1
        return self.last(h)
