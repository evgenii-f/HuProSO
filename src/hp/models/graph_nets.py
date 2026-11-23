"""A graph score network on the joints (RSGM's net="mace"; docs/mace_lite_plan.md).

The joints are the nodes, the edges are given (the kinematic tree and further pairs, both directions), and every edge
carries the rotation of its target joint relative to its source, R_ik = G_i^T G_k with G the rotations relative to
the root (for a parent and its child that is x_k). In the spirit of MACE (Batatia et al. 2022), without equivariance
(a pose prior has no rotation symmetry to keep):

    h_j   = SiLU(Linear([e_j, vec(x_j)]) + T_0(temb))                 e_j a learned embedding of the joint
    L layers:
      phi_e = MLP([vec(R_e), f_type(e)]) + T_phi(temb)               a learned function of the edge (C channels)
      A_i   = sum over the edges e = (i <- k) of phi_e * W h_k      the analog of the atomic base of ACE
      B_i   = (U A_i) * (V A_i)                                       channel-wise products: body order 3, as MACE
      h_i   = h_i + Dropout(MLP([h_i, A_i, B_i]) + T(temb))          residual update
    out_i = Q h_i                                                     3 numbers per joint, the scaled score

global_node adds a virtual node connected to every joint (as in the graph networks for molecules, Gilmer et al.
2017), so that a joint learns about the whole pose in one layer instead of one edge per layer:

    g     = SiLU(Linear(vec(x)) + T_g(temb))                         the whole pose, as the input of a dense network
    in every layer, before the update of the joints:
      g   = g + MLP([g, mean_j h_j])
      h_i = h_i + Dropout(MLP([h_i, A_i, B_i, g]) + T(temb))

temb is an MLP of the sinusoidal embedding of the noise level, as in score_nets.TimeResNet. The sum over the edges is
a matrix product with a fixed incidence matrix, so that torch.func can vmap and differentiate the network (the exact
divergence of RSGM's log_prob). The weights are shared over the edges and the joints; only the embeddings tell them
apart, so the size does not grow with the input as a dense network's first layer does.
"""
from typing import Sequence, Tuple

import torch
from torch import nn

from hp.models.score_nets import timestep_embedding


class GraphScoreNet(nn.Module):
    def __init__(self, num_joints: int, edges: Sequence[Tuple[int, int]], channels: int = 256, layers: int = 3,
                 dropout: float = 0.0, joint_embed: int = 32, edge_embed: int = 16, time_embed: int = 128,
                 global_node: bool = False):
        """edges: directed (target i, source k) pairs; each gets its own type embedding."""
        super().__init__()
        self.num_joints, self.time_embed, self.global_node = num_joints, time_embed, global_node
        targets = torch.tensor([i for i, _ in edges])
        sources = torch.tensor([k for _, k in edges])
        incidence = torch.zeros(num_joints, len(edges))
        incidence[targets, torch.arange(len(edges))] = 1.0
        self.register_buffer("targets", targets, persistent=False)
        self.register_buffer("sources", sources, persistent=False)
        self.register_buffer("incidence", incidence, persistent=False)
        c = channels
        self.joint = nn.Parameter(torch.randn(num_joints, joint_embed) * 0.1)
        self.edge_type = nn.Parameter(torch.randn(len(edges), edge_embed) * 0.1)
        self.time = nn.Sequential(nn.Linear(time_embed, c), nn.SiLU(), nn.Linear(c, c), nn.SiLU())
        self.node_in = nn.Linear(joint_embed + 9, c)
        self.time_in = nn.Linear(c, c)
        if global_node:
            self.global_in = nn.Linear(9 * num_joints, c)
            self.global_t = nn.Linear(c, c)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "phi": nn.Sequential(nn.Linear(9 + edge_embed, c), nn.SiLU(), nn.Linear(c, c)),
                "phi_t": nn.Linear(c, c),
                "w": nn.Linear(c, c, bias=False),
                "u": nn.Linear(c, c, bias=False),
                "v": nn.Linear(c, c, bias=False),
                "upd": nn.Sequential(nn.Linear((4 if global_node else 3) * c, c), nn.SiLU(), nn.Linear(c, c)),
                "upd_t": nn.Linear(c, c),
                **({"glob": nn.Sequential(nn.Linear(2 * c, c), nn.SiLU(), nn.Linear(c, c))} if global_node else {}),
            }))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c, 3)

    def forward(self, x: torch.Tensor, g: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """x [B, J, 3, 3] the rotations relative to the parents, g [B, J, 3, 3] relative to the root, level [B] the
        noise level -> [B, 3J]."""
        b, j = x.shape[0], self.num_joints
        temb = self.time(timestep_embedding(1000 * level, self.time_embed).to(x.dtype))  # [B, C]
        joint = self.joint.to(x.dtype).expand(b, -1, -1)
        h = torch.nn.functional.silu(self.node_in(torch.cat([joint, x.reshape(b, j, 9)], -1))
                                     + self.time_in(temb)[:, None])
        rel = (g[:, self.targets].transpose(-1, -2) @ g[:, self.sources]).reshape(b, -1, 9)  # [B, E, 9]
        edge = torch.cat([rel, self.edge_type.to(x.dtype).expand(b, -1, -1)], -1)
        incidence = self.incidence.to(x.dtype)
        if self.global_node:
            glob = torch.nn.functional.silu(self.global_in(x.reshape(b, 9 * j)) + self.global_t(temb))  # [B, C]
        for layer in self.layers:
            phi = layer["phi"](edge) + layer["phi_t"](temb)[:, None]  # [B, E, C]
            messages = phi * layer["w"](h)[:, self.sources]  # [B, E, C]
            a = incidence @ messages  # [B, J, C]: the sum over the edges into each joint
            products = layer["u"](a) * layer["v"](a)
            parts = [h, a, products]
            if self.global_node:
                glob = glob + layer["glob"](torch.cat([glob, h.mean(1)], -1))
                parts.append(glob[:, None].expand(-1, j, -1))
            h = h + self.dropout(layer["upd"](torch.cat(parts, -1)) + layer["upd_t"](temb)[:, None])
        return self.out(h).reshape(b, 3 * j)
