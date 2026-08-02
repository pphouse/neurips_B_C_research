"""Pooling of a (L, D) activation tensor around a variant token/residue position."""
from __future__ import annotations

import numpy as np
import torch


def pool_exact(act: torch.Tensor, pos: int) -> torch.Tensor:
    """act: (L, D). Return activation at the exact position."""
    return act[pos]


def pool_local_mean(act: torch.Tensor, pos: int, radius: int = 8) -> torch.Tensor:
    lo = max(0, pos - radius)
    hi = min(act.shape[0], pos + radius + 1)
    return act[lo:hi].mean(dim=0)


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()
