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

Every component can be switched off, for the ablation of what makes the network work (the defaults are all on):
  - time_in: "layers" (a projection of the time in every layer) or "input" (the time concatenated to the input once);
  - time_embed: "sinusoidal" (the embedding and its shared MLP) or "scalar" (the level itself, one number);
  - residual: the skip connection of the blocks (off: the layers in a row);
  - norm: LayerNorm in every layer (off: none);
  - dropout: in every layer (0: none);
  - zero_init: the last layer starts at zero (off: PyTorch's default initialization).
With everything off (time_in="input", time_embed="scalar", residual, norm and zero_init off, dropout 0) it is a plain
MLP of 1 + 2 B layers with the level at the input, as RSGM's net="mlp".
(Which level, log s or t, is RSGM's choice: its net_level.)
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
    """Linear(h) [+ Linear_t(temb)], then Norm (or nothing), SiLU and Dropout."""

    def __init__(self, width_in: int, width: int, time_width: int, norm: bool, dropout: float):
        super().__init__()
        self.dense = nn.Linear(width_in, width)
        self.dense_t = nn.Linear(time_width, width) if time_width else None
        self.norm = nn.LayerNorm(width) if norm else nn.Identity()
        self.act, self.dropout = nn.SiLU(), nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        z = self.dense(h)
        if self.dense_t is not None:
            z = z + self.dense_t(temb)
        return self.dropout(self.act(self.norm(z)))


class TimeResNet(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int = 1024, blocks: int = 2, embed: int = 256,
                 dropout: float = 0.0, time_in: str = "layers", time_embed: str = "sinusoidal", residual: bool = True,
                 norm: bool = True, zero_init: bool = True):
        super().__init__()
        if time_in not in ("layers", "input") or time_embed not in ("sinusoidal", "scalar"):
            raise ValueError(f"time_in must be 'layers' or 'input', time_embed 'sinusoidal' or 'scalar'; "
                             f"got {time_in}, {time_embed}.")
        self.embed, self.time_in, self.time_embed, self.residual = embed, time_in, time_embed, residual
        time_width = embed if time_embed == "sinusoidal" else 1
        self.time = nn.Sequential(nn.Linear(embed, embed), nn.SiLU()) if time_embed == "sinusoidal" else None
        per_layer = time_width if time_in == "layers" else 0
        self.first = _Layer(dim_in + (time_width if time_in == "input" else 0), width, per_layer, norm, dropout)
        self.blocks = nn.ModuleList(nn.ModuleList([_Layer(width, width, per_layer, norm, dropout) for _ in range(2)])
                                    for _ in range(blocks))
        self.last = nn.Linear(width, dim_out)
        if zero_init:
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

    def forward(self, features: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """features [B, dim_in], level [B] (the noise level) -> [B, dim_out]."""
        if self.time is not None:
            temb = self.time(timestep_embedding(1000 * level, self.embed).to(features.dtype))
        else:
            temb = level[:, None].to(features.dtype)
        if self.time_in == "input":
            features = torch.cat([features, temb], dim=1)
        h = self.first(features, temb)
        for block in self.blocks:
            h1 = h
            for layer in block:
                h1 = layer(h1, temb)
            h = h + h1 if self.residual else h1
        return self.last(h)


# ---- a block of clusters of joints next to the score network ----

# The 19 joints of AmassJoints4Learning by body part: the legs, the spine, the neck and head, the arms.
CLUSTERS = (("LeftUpLeg", "LeftLeg", "LeftFoot"), ("RightUpLeg", "RightLeg", "RightFoot"),
            ("Spine0", "Spine1", "Spine2"), ("Neck", "Head"),
            ("LeftShoulder", "LeftArm", "LeftElbow", "LeftHand"),
            ("RightShoulder", "RightArm", "RightElbow", "RightHand"))


class ClusterBlock(nn.Module):
    """A score term from clusters of joints, added to the output of the score network (RSGM's net_clusters):

        e_j  = MLP([x_j, emb_j, level])          one network for every joint, emb_j a learned embedding of the joint
        c_k  = mean of e_j over the joints of cluster k
        c'   = MLP([c_1 ... c_K, level])           the exchange between the clusters, K tokens out
        out_j = Linear([e_j, c'_k(j)])             3 numbers per joint, initialized to zero

    With the last layer at zero the block adds nothing at the start, so a network with it starts as the one without.
    The output [B, 3J] is in the layout of the score network's (joint-major, 3 per joint)."""

    def __init__(self, names, clusters=CLUSTERS, width: int = 128, embed: int = 16, dropout: float = 0.0):
        super().__init__()
        names = list(names)
        index = [[names.index(n) for n in cluster] for cluster in clusters]
        owner = torch.full((len(names),), -1, dtype=torch.long)
        for k, joints in enumerate(index):
            owner[joints] = k
        if (owner < 0).any() or sum(len(c) for c in index) != len(names):
            raise ValueError("every joint must be in exactly one cluster")
        member = torch.zeros(len(clusters), len(names))
        for k, joints in enumerate(index):
            member[k, joints] = 1.0 / len(joints)
        self.register_buffer("owner", owner, persistent=False)
        self.register_buffer("member", member, persistent=False)  # [K, J]: the mean over a cluster
        self.joints, self.k, self.width = len(names), len(clusters), width
        self.embedding = nn.Parameter(0.1 * torch.randn(len(names), embed))
        self.joint = nn.Sequential(nn.Linear(9 + embed + 1, width), nn.SiLU(), nn.Dropout(dropout),
                                   nn.Linear(width, width), nn.SiLU())
        self.exchange = nn.Sequential(nn.Linear(self.k * width + 1, 2 * width), nn.SiLU(), nn.Dropout(dropout),
                                      nn.Linear(2 * width, self.k * width), nn.SiLU())
        self.out = nn.Linear(2 * width, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """x [B, J, 3, 3] the local rotations, level [B] the noise level -> [B, 3J]."""
        b = x.shape[0]
        lv = level.to(x.dtype)[:, None]
        e = self.joint(torch.cat([x.reshape(b, self.joints, 9), self.embedding.to(x.dtype).expand(b, -1, -1),
                                  lv[:, None].expand(-1, self.joints, 1)], -1))  # [B, J, W]
        c = torch.einsum("kj,bjw->bkw", self.member.to(x.dtype), e)  # [B, K, W]
        c = self.exchange(torch.cat([c.reshape(b, -1), lv], -1)).reshape(b, self.k, self.width)
        return self.out(torch.cat([e, c[:, self.owner]], -1)).reshape(b, 3 * self.joints)
