"""Keypoint conditions of the models of the zoo (RSGM, PoseD-Flow): the model learns p(pose | c) for keypoints c, the
joint positions of hp.data.conditions (22 x 3, or 22 x 2). Only the network sees c: a small MLP (one hidden layer, as
the condition MLP of the HuProSO3 paper) maps the keypoints and their mask to features that the network takes as an
extra input.
  - cond [B, 22 * k] are flattened keypoints. A joint with a NaN coordinate is not observed: its keypoint is set to
    0 and its bit of the mask to 0 (the mask is an input of the condition MLP).
  - In the loss, cond may be left out: the keypoints are then computed from the poses on the fly (of_poses).
  - cond_mask = "random" hides joints at random in the loss, as the paper's training for occlusions: per pose a
    probability p ~ U(0, 1), and every joint is hidden with probability p (hide_random).
Without `condition`, cond_dim > 0 makes a generic condition vector [B, cond_dim] (no mask, no keypoints computed on the
fly).
"""
from typing import Optional

import torch
from torch import nn

from hp.data.conditions import NUM_KEYPOINTS, condition_dim, keypoints


def move(x: torch.Tensor, device, dtype) -> torch.Tensor:
    """x on `device` as `dtype`, in two steps: on mps, changing device and dtype in one .to() sometimes
    returns garbage (torch 2.13), so the dtype is changed on the CPU side of the copy."""
    if torch.device(device).type == "cpu":
        return x.to("cpu").to(dtype)
    return x.to(dtype).to(device)


def resolve(condition: Optional[str], cond_dim: int, cond_mask: str) -> int:
    """The cond_dim of the settings (set by `condition`), or raise if they do not fit together."""
    if condition is not None:
        if cond_dim not in (0, condition_dim(condition)):
            raise ValueError(f"condition='{condition}' has cond_dim {condition_dim(condition)}, not {cond_dim}.")
        cond_dim = condition_dim(condition)
    if cond_mask not in ("none", "random"):
        raise ValueError(f"Unknown cond_mask '{cond_mask}'. Use 'none' or 'random'.")
    if cond_mask != "none" and condition is None:
        raise ValueError("cond_mask hides keypoints: it needs condition='3D' or '2D'.")
    return cond_dim


def make_net(cond_dim: int, condition: Optional[str], hidden: int, features: int) -> Optional[nn.Sequential]:
    """The condition MLP: keypoints and their mask (or a generic condition vector) -> features; None if cond_dim is 0."""
    if not cond_dim:
        return None
    inputs = cond_dim + (NUM_KEYPOINTS if condition is not None else 0)
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, features))


def features(net: Optional[nn.Module], condition: Optional[str], cond: Optional[torch.Tensor], batch: int, device,
             dtype) -> torch.Tensor:
    """Features [B, F] of the condition [B, cond_dim] (NaN: joint not observed) on `device` as `dtype`; [B, 0] without
    a condition MLP."""
    if net is None:
        return torch.zeros(batch, 0, device=device, dtype=dtype)
    if condition is None:  # a generic condition vector
        return net(move(cond, device, dtype))
    c = move(cond, device, dtype).reshape(batch, NUM_KEYPOINTS, -1)
    seen = torch.isfinite(c).all(dim=-1)
    c = torch.where(seen[..., None], c, torch.zeros_like(c))
    return net(torch.cat([c.reshape(batch, -1), seen.to(dtype)], dim=1))


def of_poses(pose: torch.Tensor, cond: Optional[torch.Tensor], condition: Optional[str]) -> Optional[torch.Tensor]:
    """The condition of the loss: `cond`, or the keypoints of the poses [B, J, 3, 3] if it is None and the model has
    keypoint conditions."""
    if condition is None or cond is not None:
        return cond
    batch = pose.shape[0]
    with torch.no_grad():
        return keypoints(pose.detach().reshape(batch, -1, 3, 3), condition).reshape(batch, -1)


def hide_random(cond: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Keypoints [B, 22 * k] with joints hidden (NaN) at random: per pose p ~ U(0, 1), every joint with probability p."""
    batch = cond.shape[0]
    p = torch.rand(batch, 1, dtype=torch.float64, generator=generator)
    hidden = (torch.rand(batch, NUM_KEYPOINTS, dtype=torch.float64, generator=generator) < p).to(cond.device)
    return cond.reshape(batch, NUM_KEYPOINTS, -1).masked_fill(hidden[..., None], float("nan")).reshape(batch, -1)
