#!/usr/bin/env python3
"""Research C: train the Delta-Crosscoder on matched base/ft variant deltas and evaluate
the latent taxonomy, delta-reconstruction advantage over a standard crosscoder, and whether
fine-tune-specific latents carry the LOF signal."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cdd.crosscoder.delta_crosscoder import DeltaCCConfig, DeltaCrosscoder, delta_cc_loss
from cdd.crosscoder.model import fve
from cdd.utils.common import load_yaml, save_json, set_seed


def train_dcc(hb, hf, tr, cfg, w, dev, steps):
    D = hb.shape[1]
    m = DeltaCrosscoder(DeltaCCConfig(d=D, n_latents=cfg["n_latents"], topk=cfg["topk"])).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.get("lr", 1e-3))
    Hb, Hf = torch.tensor(hb, device=dev), torch.tensor(hf, device=dev)
    Hbt, Hft = Hb[tr], Hf[tr]
    B = min(cfg.get("batch_size", 256), len(tr))
    for step in range(steps):
        idx = torch.randint(0, len(tr), (B,), device=dev)
        out = m(Hbt[idx], Hft[idx])
        loss, parts = delta_cc_loss(out, Hbt[idx], Hft[idx], w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, Hb, Hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 0)); dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(cfg["run_dir"]); run.mkdir(parents=True, exist_ok=True)

    z = np.load(Path(cfg["matched_dir"]) / "matched.npz")
    import pandas as pd
    idx_df = pd.read_parquet(Path(cfg["matched_dir"]) / "index.parquet")
    base, ft, y = z["base"].astype(np.float32), z["ft"].astype(np.float32), z["y"]
    split = idx_df[cfg.get("split_col", "split_domain")].to_numpy()
    tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]

    # shared PCA (fit on base+ft train) so both live in the same reduced space
    both_tr = np.concatenate([base[tr], ft[tr]], 0)
    std = both_tr.std(0) + 1e-6
    npca = min(cfg.get("n_pca", 128), len(tr) - 1)
    pca = PCA(n_components=npca, whiten=True).fit(both_tr / std)
    hb = pca.transform(base / std).astype(np.float32)
    hf = pca.transform(ft / std).astype(np.float32)
    print(f"C delta-cc: n={len(base)} train={len(tr)} test={len(te)} dim={hb.shape[1]} "
          f"model-diff L2(reduced)={np.linalg.norm(hf-hb,axis=1).mean():.3f}")

    steps = cfg.get("steps", 4000)
    # Delta-Crosscoder vs standard crosscoder (delta weight 0)
    results = {}
    for name, w in [("delta_cc", dict(base=1.0, ft=1.0, delta=cfg.get("lambda_delta", 2.0))),
                    ("standard_cc", dict(base=1.0, ft=1.0, delta=0.0))]:
        m, Hb, Hf = train_dcc(hb, hf, tr, cfg, w, dev, steps)
        with torch.no_grad():
            o = m(Hb[te], Hf[te])
            fb, ff = fve(Hb[te], o["hb_hat"]), fve(Hf[te], o["hf_hat"])
            dtrue = Hf[te] - Hb[te]; dhat = o["hf_hat"] - o["hb_hat"]
            fd = fve(dtrue, dhat)
        tax = m.taxonomy()
        from collections import Counter
        results[name] = dict(fve_base=fb, fve_ft=ff, fve_delta=fd,
                             taxonomy_counts=dict(Counter(tax["cls"])))
        print(f"[{name}] FVE base={fb:.3f} ft={ff:.3f} DELTA={fd:.3f} tax={results[name]['taxonomy_counts']}")
        if name == "delta_cc":
            torch.save({"state_dict": m.state_dict(), "cfg": m.cfg.__dict__,
                        "pca_components": pca.components_, "std": std}, run / "delta_cc.pt")
            # which latents carry LOF? latent-only classifier (ft-specific + amplified)
            with torch.no_grad():
                zt = m(Hb, Hf)["z"].cpu().numpy()
            tax_arr = np.array(tax["cls"])
            for group in ["ft_specific", "amplified", "shared"]:
                cols = np.where(np.isin(tax_arr, [group]))[0]
                if len(cols) == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    results[f"lof_auroc_{group}"] = None; continue
                sc = StandardScaler().fit(zt[tr][:, cols])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(zt[tr][:, cols]), y[tr])
                p = clf.predict_proba(sc.transform(zt[te][:, cols]))[:, 1]
                results[f"lof_auroc_{group}"] = float(roc_auc_score(y[te], p))
            results["n_ft_specific"] = int((tax_arr == "ft_specific").sum())
            results["n_amplified"] = int((tax_arr == "amplified").sum())

    save_json(results, run / "eval_c.json")
    print("Saved", run / "eval_c.json")
    print("Delta-recon advantage:",
          round(results["delta_cc"]["fve_delta"] - results["standard_cc"]["fve_delta"], 3))


if __name__ == "__main__":
    main()
