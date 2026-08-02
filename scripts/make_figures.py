#!/usr/bin/env python3
"""Generate paper figures from evaluation outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"cc": "#2274A5", "base": "#B0B0B0", "priv": "#E83151", "shared": "#32936F"}


def fig_dms_bar(evalj, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, split in zip(axes, ["split_position", "split_domain"]):
        if split not in evalj:
            ax.set_visible(False); continue
        d = evalj[split]["dms"]
        order = ["dna_only", "prot_only", "cca", "concat", "shared_code"]
        labels = ["Evo2\nΔ", "ESM2\nΔ", "CCA", "concat", "shared\n(ours)"]
        vals = [d.get(k, np.nan) for k in order]
        colors = [C["base"], C["base"], C["base"], C["base"], C["shared"]]
        ax.bar(labels, vals, color=colors)
        ext = evalj[split].get("dms_external", {})
        if "cadd" in ext:
            ax.axhline(ext["cadd"], ls="--", c="k", lw=1, label=f"CADD {ext['cadd']:.2f}")
            ax.legend(fontsize=8)
        ax.set_title(f"DMS Spearman ({split.replace('split_','')}-disjoint)")
        ax.set_ylim(0, max(0.6, max([v for v in vals if not np.isnan(v)] + [0]) * 1.2))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_retrieval(evalj, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    split = "split_position"
    cc = evalj[split]["retrieval_crosscoder"]
    cca = evalj[split]["retrieval_cca"]
    ks = ["R@1", "R@5", "R@10"]
    x = np.arange(len(ks)); w = 0.35
    ax.bar(x - w/2, [cc[k] for k in ks], w, label="shared crosscoder", color=C["cc"])
    ax.bar(x + w/2, [cca[k] for k in ks], w, label="CCA", color=C["base"])
    n = cc["n"]
    ax.axhline(1/n, ls=":", c="k", lw=1, label=f"chance (1/{n})")
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("Recall")
    ax.set_title("Cross-modal variant retrieval"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_train(run_dir, out):
    log = json.load(open(Path(run_dir) / "train_log.json"))
    df = pd.DataFrame(log)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(df.step, df.fve_dna, label="FVE DNA", color=C["cc"])
    ax.plot(df.step, df.fve_prot, label="FVE protein", color=C["priv"])
    ax.set_xlabel("step"); ax.set_ylabel("FVE (test)"); ax.legend(); ax.set_ylim(0, 1)
    ax.set_title("Reconstruction quality")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    evalj = json.load(open(Path(args.run_dir) / "eval_b.json"))
    fig_dms_bar(evalj, out / "fig_dms.png")
    fig_retrieval(evalj, out / "fig_retrieval.png")
    fig_train(args.run_dir, out / "fig_train.png")
    print("figures written to", out)


if __name__ == "__main__":
    main()
