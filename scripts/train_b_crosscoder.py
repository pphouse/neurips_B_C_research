#!/usr/bin/env python3
"""Train the Shared-Private Variant-Delta Crosscoder on BRCA1 paired deltas."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import (
    CrosscoderConfig, SharedPrivateCrosscoder, crosscoder_loss, fve,
)
from cdd.utils.common import load_yaml, save_json, set_seed, RunContext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(cfg["run_dir"])
    RunContext(run_dir, cfg, seed).save()

    pd_ = load_paired(
        cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
        pooling=cfg.get("pooling", "exact"), norm_split_col=cfg.get("split_col", "split_position"),
        n_pca=cfg.get("n_pca"),
    )
    split = pd_.meta[cfg.get("split_col", "split_position")].to_numpy()
    tr = split == "train"
    te = split == "test"
    print(f"Paired N={len(pd_.dna)}  train={tr.sum()} test={te.sum()}")

    Xd = torch.tensor(pd_.dna, device=dev)
    Xp = torch.tensor(pd_.prot, device=dev)
    Xd_tr, Xp_tr = Xd[tr], Xp[tr]
    Xd_te, Xp_te = Xd[te], Xp[te]

    ccfg = CrosscoderConfig(
        d_dna=Xd.shape[1], d_prot=Xp.shape[1],
        k_shared=cfg["k_shared"], k_private=cfg["k_private"],
        topk_shared=cfg["topk_shared"], topk_private=cfg["topk_private"],
    )
    model = SharedPrivateCrosscoder(ccfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    w = cfg["loss_weights"]

    B = cfg.get("batch_size", 256)
    steps = cfg.get("steps", 4000)
    warmup = int(cfg.get("warmup_frac", 0.3) * steps)  # rec-only warmup
    ntr = Xd_tr.shape[0]
    t0 = time.time()
    log = []
    for step in range(steps):
        idx = torch.randint(0, ntr, (min(B, ntr),), device=dev)
        out = model(Xd_tr[idx], Xp_tr[idx])
        # ramp alignment/contrast/orth in after reconstruction warmup
        ramp = min(1.0, max(0.0, (step - warmup) / max(1, 0.3 * steps)))
        wnow = dict(w)
        for k in ("align", "orth"):
            wnow[k] = w[k] * ramp
        loss, parts = crosscoder_loss(out, Xd_tr[idx], Xp_tr[idx], wnow)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                o = model(Xd_te, Xp_te)
                otr = model(Xd_tr, Xp_tr)
                fd, fp = fve(Xd_te, o["xhat_d"]), fve(Xp_te, o["xhat_p"])
                fdt = fve(Xd_tr, otr["xhat_d"])
                l0d = (o["zs_d"] > 0).float().sum(1).mean().item()
                dead = float((otr["zs_d"].sum(0) == 0).float().mean().item())
            model.train()
            rec = dict(step=step, loss=loss.item(), fve_dna=fd, fve_dna_train=fdt,
                       fve_prot=fp, l0_shared=l0d, dead_shared=dead, ramp=ramp, **parts)
            log.append(rec)
            print(f"  step {step}: loss={loss.item():.3f} FVE_dna={fd:.3f}(tr {fdt:.3f}) "
                  f"FVE_prot={fp:.3f} L0s={l0d:.1f} dead={dead:.2f} align={parts['align']:.3f} "
                  f"contrast={parts['contrast']:.3f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "cfg": ccfg.__dict__,
                "dna_std": pd_.dna_std, "prot_std": pd_.prot_std},
               run_dir / "crosscoder.pt")
    save_json(log, run_dir / "train_log.json")
    print(f"Done in {(time.time()-t0)/60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
