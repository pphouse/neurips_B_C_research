#!/usr/bin/env python3
"""Interpretability: shared vs private latent enrichment heatmap + top-variant tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder
from cdd.utils.common import load_yaml

plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})


def auroc_matrix(z, label_dict):
    """z: (N,F) codes; returns (F, n_labels) AUROC matrix and label names."""
    names = list(label_dict)
    M = np.full((z.shape[1], len(names)), np.nan)
    for j, nm in enumerate(names):
        y = label_dict[nm]
        keep = ~np.isnan(y)
        if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
            continue
        for f in range(z.shape[1]):
            if (z[keep, f] > 0).sum() < 5:
                continue
            try:
                M[f, j] = roc_auc_score(y[keep], z[keep, f])
            except Exception:
                pass
    return M, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
                      pooling=cfg.get("pooling", "exact"),
                      norm_split_col=cfg.get("split_col", "split_position"), n_pca=cfg.get("n_pca"))
    meta = pdd.meta
    ck = torch.load(Path(args.run_dir) / "crosscoder.pt", map_location=dev, weights_only=False)
    model = SharedPrivateCrosscoder(CrosscoderConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    a = model.encode_all(torch.tensor(pdd.dna, device=dev), torch.tensor(pdd.prot, device=dev))
    zs = a["shared_dna"].cpu().numpy()
    zpd = a["priv_dna"].cpu().numpy()
    zpp = a["priv_prot"].cpu().numpy()

    labels = {
        "RING": (meta["domain"] == "RING").astype(float).to_numpy(),
        "BRCT": (meta["domain"] == "BRCT").astype(float).to_numpy(),
        "LOF": (meta["func_class"] == "LOF").astype(float).to_numpy(),
        "functional": (meta["func_class"] == "FUNC").astype(float).to_numpy(),
    }
    Ms, names = auroc_matrix(zs, labels)
    Mpd, _ = auroc_matrix(zpd, labels)
    Mpp, _ = auroc_matrix(zpp, labels)

    # keep active latents, sort by max |AUROC-0.5|
    def top_rows(M, k=20):
        score = np.nanmax(np.abs(M - 0.5), 1)
        idx = np.argsort(-np.nan_to_num(score))[:k]
        return idx

    fig, axes = plt.subplots(1, 3, figsize=(9, 4.2))
    for ax, M, title in zip(axes, [Ms, Mpd, Mpp],
                            ["shared (DNA)", "DNA-private", "protein-private"]):
        idx = top_rows(M)
        im = ax.imshow(np.nan_to_num(M[idx], nan=0.5), aspect="auto", cmap="RdBu_r",
                       vmin=0.2, vmax=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks([]); ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025, label="latent AUROC")
    fig.suptitle("Per-latent annotation enrichment (top 20 latents each)")
    fig.savefig(out / "fig_interp.png", dpi=150)
    fig.savefig(out / "fig_interp.pdf")
    print("wrote", out / "fig_interp.png")

    # top-activating variants for the most domain/LOF-selective shared latents
    rows = []
    for j, nm in enumerate(names):
        f = int(np.nanargmax(Ms[:, j])) if np.isfinite(Ms[:, j]).any() else -1
        if f < 0:
            continue
        order = np.argsort(-zs[:, f])[:8]
        for rank, i in enumerate(order):
            r = meta.iloc[i]
            rows.append(dict(annotation=nm, latent=f, auroc=round(float(Ms[f, j]), 3),
                             rank=rank, variant=r["variant_id"], aa=f"{r['aa_ref']}{int(r['aa_pos'])}{r['aa_alt']}",
                             domain=r["domain"], func_class=r["func_class"], act=round(float(zs[i, f]), 3)))
    pd.DataFrame(rows).to_csv(Path(args.run_dir) / "top_latent_variants.csv", index=False)
    print("wrote", Path(args.run_dir) / "top_latent_variants.csv")


if __name__ == "__main__":
    main()
