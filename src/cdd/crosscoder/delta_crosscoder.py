"""Delta-Crosscoder for model diffing (Research C): diff base vs. fine-tuned Evo2.

A BatchTopK crosscoder jointly encodes matched activations (h_base, h_ft) of the SAME
input through the two models, with a shared latent set and per-model decoders. Latents are
classified by their decoder-norm ratio (shared / base-specific / ft-specific / amplified),
and a delta-reconstruction term emphasizes the base->ft difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import batch_topk, nmse, fve


@dataclass
class DeltaCCConfig:
    d: int = 4096
    n_latents: int = 256
    topk: int = 32


class DeltaCrosscoder(nn.Module):
    def __init__(self, cfg: DeltaCCConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Linear(2 * cfg.d, cfg.n_latents)
        self.dec_base = nn.Linear(cfg.n_latents, cfg.d, bias=False)
        self.dec_ft = nn.Linear(cfg.n_latents, cfg.d, bias=False)
        self.b_base = nn.Parameter(torch.zeros(cfg.d))
        self.b_ft = nn.Parameter(torch.zeros(cfg.d))
        with torch.no_grad():
            for d in (self.dec_base, self.dec_ft):
                d.weight.data = F.normalize(d.weight.data, dim=0)

    def encode(self, hb, hf):
        x = torch.cat([hb - self.b_base, hf - self.b_ft], dim=-1)
        return F.relu(self.enc(x))

    def forward(self, hb, hf):
        z = self.encode(hb, hf)
        zt = batch_topk(z, self.cfg.topk)
        hb_hat = self.dec_base(zt) + self.b_base
        hf_hat = self.dec_ft(zt) + self.b_ft
        return dict(z=zt, z_pre=z, hb_hat=hb_hat, hf_hat=hf_hat)

    @torch.no_grad()
    def decoder_norms(self):
        rb = self.dec_base.weight.norm(dim=0)   # (F,)
        rf = self.dec_ft.weight.norm(dim=0)
        return rb, rf

    @torch.no_grad()
    def taxonomy(self, tol=0.3):
        """Classify each latent by base/ft decoder norm.
        ft-specific: rb small, rf large; base-specific: reverse; shared: both large."""
        rb, rf = self.decoder_norms()
        rb = rb.cpu().numpy(); rf = rf.cpu().numpy()
        import numpy as np
        ratio = (rf + 1e-6) / (rb + 1e-6)
        cls = np.full(len(rb), "shared", dtype=object)
        cls[ratio > 1 / tol] = "ft_specific"
        cls[ratio < tol] = "base_specific"
        # amplified: shared but ft notably larger
        amp = (ratio > 1.3) & (ratio <= 1 / tol)
        cls[amp] = "amplified"
        return dict(cls=cls.tolist(), rb=rb.tolist(), rf=rf.tolist(), ratio=ratio.tolist())


def delta_cc_loss(out, hb, hf, w):
    l_base = nmse(hb, out["hb_hat"])
    l_ft = nmse(hf, out["hf_hat"])
    delta_true = hf - hb
    delta_hat = out["hf_hat"] - out["hb_hat"]
    l_delta = nmse(delta_true, delta_hat)
    total = w["base"] * l_base + w["ft"] * l_ft + w["delta"] * l_delta
    return total, dict(base=l_base.item(), ft=l_ft.item(), delta=l_delta.item())
