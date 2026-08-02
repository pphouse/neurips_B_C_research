"""Shared-Private Variant-Delta Crosscoder (Research B).

For a missense variant we have paired deltas Δh_DNA (Evo2) and Δh_PROT (ESM-2).
Each modality has a *shared* encoder/decoder and a *private* encoder/decoder:

    Δh_m -> E_m_shared -> z_m_shared ;  E_m_private -> z_m_private
    Δh_m_hat = D_m_shared(z_m_shared) + D_m_private(z_m_private)

Shared codes are aligned across modalities for the same variant (alignment +
contrastive losses); private codes are decorrelated from shared codes. Sparsity is
enforced per code-block with BatchTopK.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def batch_topk(z: torch.Tensor, k: int) -> torch.Tensor:
    """BatchTopK (Bussmann et al. 2024): keep the k*B largest activations across the
    whole batch, giving an average of k active units per sample. Straight-through:
    magnitudes of surviving units are preserved; others set to 0.
    z: (B, F) nonnegative pre-activations. Returns sparsified (B, F)."""
    B, Fdim = z.shape
    if k >= Fdim:
        return z
    flat = z.reshape(-1)
    n_keep = max(1, k * B)
    n_keep = min(n_keep, flat.numel())
    thresh = torch.topk(flat, n_keep, sorted=False).values.min()
    mask = (z >= thresh).float()
    return z * mask


@dataclass
class CrosscoderConfig:
    d_dna: int = 4096
    d_prot: int = 1280
    k_shared: int = 64        # latent width of shared space
    k_private: int = 64       # latent width of each private space
    topk_shared: int = 16     # BatchTopK L0 for shared
    topk_private: int = 16    # BatchTopK L0 for private
    tie_shared_decoder_norm: bool = True


class Branch(nn.Module):
    """One modality's shared+private encoders and decoders."""

    def __init__(self, d_in: int, k_shared: int, k_private: int):
        super().__init__()
        self.enc_shared = nn.Linear(d_in, k_shared)
        self.enc_private = nn.Linear(d_in, k_private)
        self.dec_shared = nn.Linear(k_shared, d_in, bias=False)
        self.dec_private = nn.Linear(k_private, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))
        # unit-norm decoder columns (standard SAE init)
        with torch.no_grad():
            for dec in (self.dec_shared, self.dec_private):
                dec.weight.data = F.normalize(dec.weight.data, dim=0)

    def encode(self, x):
        xs = x - self.bias
        zs = F.relu(self.enc_shared(xs))
        zp = F.relu(self.enc_private(xs))
        return zs, zp

    def decode(self, zs, zp):
        return self.dec_shared(zs) + self.dec_private(zp) + self.bias


class SharedPrivateCrosscoder(nn.Module):
    def __init__(self, cfg: CrosscoderConfig):
        super().__init__()
        self.cfg = cfg
        self.dna = Branch(cfg.d_dna, cfg.k_shared, cfg.k_private)
        self.prot = Branch(cfg.d_prot, cfg.k_shared, cfg.k_private)

    def forward(self, x_dna, x_prot):
        zs_d, zp_d = self.dna.encode(x_dna)
        zs_p, zp_p = self.prot.encode(x_prot)
        # sparsify
        zs_d_t = batch_topk(zs_d, self.cfg.topk_shared)
        zs_p_t = batch_topk(zs_p, self.cfg.topk_shared)
        zp_d_t = batch_topk(zp_d, self.cfg.topk_private)
        zp_p_t = batch_topk(zp_p, self.cfg.topk_private)
        xhat_d = self.dna.decode(zs_d_t, zp_d_t)
        xhat_p = self.prot.decode(zs_p_t, zp_p_t)
        return dict(
            zs_d=zs_d_t, zs_p=zs_p_t, zp_d=zp_d_t, zp_p=zp_p_t,
            zs_d_pre=zs_d, zs_p_pre=zs_p,
            xhat_d=xhat_d, xhat_p=xhat_p,
        )

    @torch.no_grad()
    def encode_shared(self, x_dna, x_prot, sparse: bool = True):
        """Return shared codes for both modalities. sparse=True applies BatchTopK
        (interpretable feature activations); sparse=False returns the dense ReLU codes
        (used for retrieval / probing, on which the alignment loss operates)."""
        zs_d, _ = self.dna.encode(x_dna)
        zs_p, _ = self.prot.encode(x_prot)
        if sparse:
            return batch_topk(zs_d, self.cfg.topk_shared), batch_topk(zs_p, self.cfg.topk_shared)
        return zs_d, zs_p

    @torch.no_grad()
    def encode_all(self, x_dna, x_prot):
        """Dense shared + sparse shared + sparse private codes for evaluation."""
        zs_d, zp_d = self.dna.encode(x_dna)
        zs_p, zp_p = self.prot.encode(x_prot)
        return dict(
            shared_dna_dense=zs_d, shared_prot_dense=zs_p,
            shared_dna=batch_topk(zs_d, self.cfg.topk_shared),
            shared_prot=batch_topk(zs_p, self.cfg.topk_shared),
            priv_dna=batch_topk(zp_d, self.cfg.topk_private),
            priv_prot=batch_topk(zp_p, self.cfg.topk_private),
        )


def nmse(x, xhat):
    """Normalized MSE: ||x-xhat||^2 / ||x||^2 averaged over batch."""
    num = ((x - xhat) ** 2).sum(-1)
    den = (x ** 2).sum(-1).clamp_min(1e-8)
    return (num / den).mean()


def fve(x, xhat):
    """Fraction of variance explained."""
    res = ((x - xhat) ** 2).sum()
    tot = ((x - x.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
    return (1 - res / tot).item()


def info_nce(za, zb, temp: float = 0.1):
    """Symmetric InfoNCE: row i of za matches row i of zb (same variant)."""
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    logits = za @ zb.t() / temp
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def orthogonality(zs, zp):
    """Cross-covariance penalty between shared and private codes (decorrelate)."""
    zs = zs - zs.mean(0, keepdim=True)
    zp = zp - zp.mean(0, keepdim=True)
    B = zs.size(0)
    cov = (zs.t() @ zp) / B  # (k_s, k_p)
    return (cov ** 2).mean()


def crosscoder_loss(out, x_dna, x_prot, w):
    l_rec = nmse(x_dna, out["xhat_d"]) + nmse(x_prot, out["xhat_p"])
    l_align = F.mse_loss(
        F.normalize(out["zs_d_pre"], dim=-1), F.normalize(out["zs_p_pre"], dim=-1)
    )
    l_contrast = info_nce(out["zs_d_pre"], out["zs_p_pre"], temp=w.get("temp", 0.1))
    l_orth = orthogonality(out["zs_d"], out["zp_d"]) + orthogonality(out["zs_p"], out["zp_p"])
    total = (
        w["rec"] * l_rec
        + w["align"] * l_align
        + w["contrast"] * l_contrast
        + w["orth"] * l_orth
    )
    return total, dict(rec=l_rec.item(), align=l_align.item(),
                       contrast=l_contrast.item(), orth=l_orth.item())
