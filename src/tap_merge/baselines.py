"""Conventional data-free merging controls. Named optimizer papers are not impersonated."""
import torch
from .linalg import top_mask


def average(deltas):
    return torch.stack(deltas).mean(dim=0)


def ties(deltas, density=0.2):
    """Trim, elect weighted-magnitude sign, average nonzero sign-aligned values.

    Operates on a flattened whole-model task vector; caller controls merge scale.
    """
    d = torch.stack([d * top_mask(d.abs(), density) for d in deltas])
    sign = torch.sign(d.sum(dim=0))
    aligned = (torch.sign(d) == sign) & (d != 0)
    return (d * aligned).sum(dim=0) / aligned.sum(dim=0).clamp_min(1)


def dare_ties(deltas, density=0.2, drop_rate=0.5, seed=42):
    if not 0 <= drop_rate < 1:
        raise ValueError("DARE drop_rate must be in [0,1)")
    rng = torch.Generator().manual_seed(seed)
    dropped = [d * (torch.rand(d.shape, generator=rng) >= drop_rate) / (1 - drop_rate) for d in deltas]
    return ties(dropped, density)
