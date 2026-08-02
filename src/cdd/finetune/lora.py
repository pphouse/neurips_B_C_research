"""Minimal LoRA for injecting trainable low-rank adapters into Evo2 Linear modules.

Evo2 (vortex StripedHyena) exposes standard nn.Linear submodules
(blocks.N.mlp.l1/l2/l3, blocks.N.projections, blocks.N.out_filter_dense, ...). We wrap
selected ones with a frozen base + trainable low-rank update, so the backbone stays frozen
and only the adapters (and a small head) train.
"""
from __future__ import annotations

import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora = self.drop(x) @ self.A.t() @ self.B.t()
        return out + self.scaling * lora.to(out.dtype)


def inject_lora(model: nn.Module, target_regex: str, r=8, alpha=16, dropout=0.0):
    """Wrap every nn.Linear whose qualified name matches target_regex with LoRA.
    Returns list of wrapped module names."""
    pat = re.compile(target_regex)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and pat.search(full):
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped.append(full)
    return wrapped


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if (".A" in n or ".B" in n) and p.requires_grad]
