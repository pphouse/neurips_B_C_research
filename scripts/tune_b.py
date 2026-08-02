#!/usr/bin/env python3
"""Fast sweep for a crosscoder recipe whose alignment embedding is competitive with CCA."""
import itertools
import numpy as np
import torch

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder, crosscoder_loss, fve
from cdd.eval.probes import retrieval_recall, ridge_spearman

dev = "cuda"
pdd = load_paired("outputs/act/evo2", "outputs/act/esm", "blocks.24.mlp.l3", "L33",
                  pooling="local", n_pca=128)
m = pdd.meta
tr = (m.split_position == "train").to_numpy(); te = (m.split_position == "test").to_numpy()
y = m.dms_score.to_numpy(); keep = ~np.isnan(y)
Xd = torch.tensor(pdd.dna, device=dev); Xp = torch.tensor(pdd.prot, device=dev)
print("CCA baseline (known): R@1~0.235 R@10~? DMS~0.282", flush=True)


def run(ks, da, contrast, steps=4000):
    torch.manual_seed(0)
    cfg = CrosscoderConfig(d_dna=Xd.shape[1], d_prot=Xp.shape[1], k_shared=ks, k_private=96,
                           topk_shared=min(24, ks), topk_private=24, d_align=da)
    mo = SharedPrivateCrosscoder(cfg).to(dev)
    opt = torch.optim.Adam(mo.parameters(), 1e-3)
    Xdt, Xpt = Xd[tr], Xp[tr]; n = Xdt.shape[0]; warm = int(0.2 * steps)
    for s in range(steps):
        idx = torch.randint(0, n, (256,), device=dev)
        out = mo(Xdt[idx], Xpt[idx])
        ramp = min(1.0, max(0.0, (s - warm) / (0.3 * steps)))
        w = dict(rec=1.0, align=0.05 * ramp, contrast=contrast, orth=0.1 * ramp, temp=0.1)
        loss, _ = crosscoder_loss(out, Xdt[idx], Xpt[idx], w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mo.parameters(), 1.0); opt.step()
    mo.eval()
    with torch.no_grad():
        a = mo.encode_all(Xd, Xp)
        ad = a["align_dna"].cpu().numpy(); ap = a["align_prot"].cpu().numpy()
        o = mo(Xd[te], Xp[te]); fd = fve(Xd[te], o["xhat_d"])
    r = retrieval_recall(ad[te], ap[te]); sh = np.concatenate([ad, ap], 1)
    sd, _ = ridge_spearman(sh[tr & keep], y[tr & keep], sh[te & keep], y[te & keep])
    print(f"ks={ks} da={da} c={contrast}: R@1={r['R@1']:.3f} R@10={r['R@10']:.3f} "
          f"MRR={r['MRR']:.3f} DMS={sd:.3f} FVE={fd:.3f}", flush=True)


for ks, da, c in itertools.product([16, 32], [32, 64], [2.0, 5.0]):
    run(ks, da, c)
print("SWEEP_DONE", flush=True)
