#!/usr/bin/env python3
"""Train the crosscoder over several seeds and report mean+-std for the headline metrics,
against CCA/PLS/concat baselines. Saves the seed-0 model as the canonical run for
interpretability and causal analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder, crosscoder_loss, fve
from cdd.eval.probes import retrieval_recall, ridge_spearman, cca_transform
from cdd.utils.common import load_yaml, save_json, set_seed


def train_one(Xd, Xp, tr, cfg, seed, dev):
    set_seed(seed)
    ccfg = CrosscoderConfig(d_dna=Xd.shape[1], d_prot=Xp.shape[1], k_shared=cfg["k_shared"],
                            k_private=cfg["k_private"], topk_shared=cfg["topk_shared"],
                            topk_private=cfg["topk_private"], d_align=cfg["d_align"])
    mo = SharedPrivateCrosscoder(ccfg).to(dev)
    opt = torch.optim.Adam(mo.parameters(), lr=cfg.get("lr", 1e-3))
    Xdt, Xpt = Xd[tr], Xp[tr]
    n = Xdt.shape[0]
    steps = cfg.get("steps", 4000)
    warm = int(cfg.get("warmup_frac", 0.2) * steps)
    W = cfg["loss_weights"]
    for s in range(steps):
        idx = torch.randint(0, n, (min(cfg.get("batch_size", 256), n),), device=dev)
        out = mo(Xdt[idx], Xpt[idx])
        ramp = min(1.0, max(0.0, (s - warm) / max(1, 0.3 * steps)))
        w = dict(W); w["align"] = W["align"] * ramp; w["orth"] = W["orth"] * ramp
        loss, _ = crosscoder_loss(out, Xdt[idx], Xpt[idx], w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mo.parameters(), 1.0); opt.step()
    mo.eval()
    return mo, ccfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dev = "cuda"
    run = Path(cfg["run_dir"]); run.mkdir(parents=True, exist_ok=True)
    sc = cfg.get("split_col", "split_position")

    pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
                      pooling=cfg.get("pooling", "local"), norm_split_col=sc, n_pca=cfg.get("n_pca"))
    m = pdd.meta
    tr = (m[sc] == "train").to_numpy(); te = (m[sc] == "test").to_numpy()
    y = m["dms_score"].to_numpy(); keep = ~np.isnan(y)
    Xd = torch.tensor(pdd.dna, device=dev); Xp = torch.tensor(pdd.prot, device=dev)

    # baselines (deterministic)
    base = {}
    _, cca_tr, ca_te, cb_te = cca_transform(pdd.dna[tr], pdd.prot[tr], pdd.dna[te], pdd.prot[te],
                                            n_comp=min(32, cfg["k_shared"] * 2))
    base["retrieval_cca"] = retrieval_recall(ca_te, cb_te)
    base["dms_dna"] = ridge_spearman(pdd.dna[tr & keep], y[tr & keep], pdd.dna[te & keep], y[te & keep])[0]
    base["dms_prot"] = ridge_spearman(pdd.prot[tr & keep], y[tr & keep], pdd.prot[te & keep], y[te & keep])[0]
    cc = np.concatenate([pdd.dna, pdd.prot], 1)
    base["dms_concat"] = ridge_spearman(cc[tr & keep], y[tr & keep], cc[te & keep], y[te & keep])[0]
    base["dms_cca"] = ridge_spearman(cca_tr[keep[tr]], y[tr & keep], ca_te[keep[te]], y[te & keep])[0]

    per_seed = []
    for seed in range(args.seeds):
        mo, ccfg = train_one(Xd, Xp, tr, cfg, seed, dev)
        with torch.no_grad():
            a = mo.encode_all(Xd, Xp)
            ad = a["align_dna"].cpu().numpy(); ap = a["align_prot"].cpu().numpy()
            o = mo(Xd[te], Xp[te]); fd = fve(Xd[te], o["xhat_d"]); fp = fve(Xp[te], o["xhat_p"])
        r = retrieval_recall(ad[te], ap[te])
        sh = np.concatenate([ad, ap], 1)
        dms = ridge_spearman(sh[tr & keep], y[tr & keep], sh[te & keep], y[te & keep])[0]
        per_seed.append(dict(seed=seed, R1=r["R@1"], R10=r["R@10"], MRR=r["MRR"],
                             dms=dms, fve_dna=fd, fve_prot=fp))
        if seed == 0:
            torch.save({"state_dict": mo.state_dict(), "cfg": ccfg.__dict__,
                        "dna_std": pdd.dna_std, "prot_std": pdd.prot_std}, run / "crosscoder.pt")
        print(f"seed {seed}: R@1={r['R@1']:.3f} R@10={r['R@10']:.3f} DMS={dms:.3f} FVE={fd:.3f}", flush=True)

    df = pd.DataFrame(per_seed)
    agg = {k: [round(float(df[k].mean()), 4), round(float(df[k].std()), 4)] for k in
           ["R1", "R10", "MRR", "dms", "fve_dna", "fve_prot"]}
    out = dict(split=sc, n_test=int(te.sum()), seeds=args.seeds, crosscoder=agg,
               baselines=base, per_seed=per_seed,
               dna_layer=cfg["dna_layer"], prot_layer=cfg["prot_layer"])
    save_json(out, run / "seeds_summary.json")
    print("\n=== SUMMARY ({} split, n_test={}) ===".format(sc, int(te.sum())))
    print(f"Crosscoder  R@1 {agg['R1'][0]:.3f}±{agg['R1'][1]:.3f}  R@10 {agg['R10'][0]:.3f}  "
          f"DMS {agg['dms'][0]:.3f}±{agg['dms'][1]:.3f}  FVE {agg['fve_dna'][0]:.2f}")
    print(f"CCA         R@1 {base['retrieval_cca']['R@1']:.3f}  R@10 {base['retrieval_cca']['R@10']:.3f}  "
          f"DMS {base['dms_cca']:.3f}")
    print(f"baselines DMS: dna {base['dms_dna']:.3f}  prot {base['dms_prot']:.3f}  concat {base['dms_concat']:.3f}")


if __name__ == "__main__":
    main()
