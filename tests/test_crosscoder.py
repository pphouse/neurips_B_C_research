"""Synthetic tests for the crosscoder (CPU, fast)."""
import torch

from cdd.crosscoder.model import (
    CrosscoderConfig, SharedPrivateCrosscoder, batch_topk, crosscoder_loss, fve,
)


def test_batch_topk_l0():
    z = torch.rand(32, 100).abs()
    zt = batch_topk(z, k=8)
    avg_l0 = (zt > 0).float().sum(1).mean().item()
    # average L0 across batch should be ~= k
    assert 6 <= avg_l0 <= 10
    # sparsified values are a subset of originals
    assert torch.all((zt == 0) | (zt == z))


def test_forward_shapes_and_recon_improves():
    torch.manual_seed(0)
    cfg = CrosscoderConfig(d_dna=64, d_prot=32, k_shared=32, k_private=32,
                           topk_shared=8, topk_private=8)
    m = SharedPrivateCrosscoder(cfg)
    # synthetic paired data with genuine shared factor
    B = 256
    shared = torch.randn(B, 8)
    x_dna = shared @ torch.randn(8, 64) + 0.1 * torch.randn(B, 64)
    x_prot = shared @ torch.randn(8, 32) + 0.1 * torch.randn(B, 32)
    w = dict(rec=1.0, align=0.5, contrast=0.2, orth=0.1, temp=0.1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    out0 = m(x_dna, x_prot)
    f0 = fve(x_dna, out0["xhat_d"])
    for _ in range(300):
        opt.zero_grad()
        out = m(x_dna, x_prot)
        loss, _ = crosscoder_loss(out, x_dna, x_prot, w)
        loss.backward()
        opt.step()
    out1 = m(x_dna, x_prot)
    f1 = fve(x_dna, out1["xhat_d"])
    assert out1["zs_d"].shape == (B, 32)
    assert f1 > f0
    assert f1 > 0.5  # should explain most DNA variance


def test_shared_alignment_learns():
    """After training, shared codes should retrieve the paired variant better than chance."""
    torch.manual_seed(1)
    cfg = CrosscoderConfig(d_dna=64, d_prot=32, k_shared=32, k_private=16,
                           topk_shared=8, topk_private=6)
    m = SharedPrivateCrosscoder(cfg)
    B = 128
    shared = torch.randn(B, 8)
    x_dna = shared @ torch.randn(8, 64) + 0.1 * torch.randn(B, 64)
    x_prot = shared @ torch.randn(8, 32) + 0.1 * torch.randn(B, 32)
    w = dict(rec=1.0, align=1.0, contrast=1.0, orth=0.1, temp=0.1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        out = m(x_dna, x_prot)
        loss, _ = crosscoder_loss(out, x_dna, x_prot, w)
        loss.backward()
        opt.step()
    zd, zp = m.encode_shared(x_dna, x_prot)
    zd = torch.nn.functional.normalize(zd, dim=-1)
    zp = torch.nn.functional.normalize(zp, dim=-1)
    sim = zd @ zp.t()
    r1 = (sim.argmax(1) == torch.arange(B)).float().mean().item()
    assert r1 > 0.5  # far above 1/128 chance
