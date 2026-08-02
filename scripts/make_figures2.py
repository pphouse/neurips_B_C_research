#!/usr/bin/env python3
"""Second results figure: cross-gene generalization, the shared functional axis (with controls),
and the biological specialization of the DNA-private code."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})
CC = "#2274A5"; BASE = "#B0B0B0"; SH = "#32936F"; PD = "#E8A13A"; PP = "#E83151"


def L(p):
    return json.load(open(p))


def main():
    mg = L("outputs/multigene/eval_mg.json")
    axis = L("outputs/b_mvp/shared_axis_control.json")
    spl = L("outputs/b_mvp/splice_probe.json")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

    # (a) cross-gene: retrieval R@10 (unseen genes) + gene-disjoint AUROC
    labels = ["CCA", "deep\nCCA", "crosscoder"]
    r10 = [mg["retrieval_cca"]["R10"], mg["retrieval_deepcca"]["R10"], mg["retrieval_crosscoder"]["R10"]]
    x = np.arange(3)
    b = ax[0].bar(x, r10, 0.6, color=[BASE, CC, SH])
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("Recall@10 (unseen genes)")
    ax[0].set_title("(a) Cross-gene retrieval (15 genes)")
    ax[0].axhline(0, color="k", lw=0.5)
    # AUROC annotation
    au = mg["clinvar_auroc"]
    ax[0].text(0.5, 0.92, f"gene-disjoint ClinVar AUROC:\nshared {au['shared'][0]:.3f}  (CCA {au['cca']:.3f},\nEvo2 {au['dna']:.3f}, ESM {au['prot']:.3f})",
               transform=ax[0].transAxes, ha="center", va="top", fontsize=7.5,
               bbox=dict(boxstyle="round", fc="#f4f4f4", ec="none"))

    # (b) shared functional axis cosines
    names = ["functional\n(fn,fn)", "domain\n(dom,dom)", "cross\n(fn,dom)", "random"]
    vals = [axis["cos_dms"], axis["cos_domain"], axis["cos_cross"], axis["cos_random_mean"]]
    cols = [SH, CC, PP, BASE]
    ax[1].bar(np.arange(4), vals, 0.6, color=cols)
    ax[1].axhline(axis["cos_random_p95"], ls="--", c="k", lw=1, label=f"random 95th pctile |cos| {axis['cos_random_p95']:.2f}")
    ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(names); ax[1].set_ylabel("cross-modal cosine")
    ax[1].set_title("(b) Shared functional axis"); ax[1].legend(fontsize=7.5); ax[1].axhline(0, color="k", lw=0.5)

    # (c) DNA-private code activation by consequence class
    cls = ["synonymous\n(silent)", "missense", "splice"]
    act = [spl["dna_priv_l2_synon"] if "dna_priv_l2_synon" in spl else spl.get("dna_priv_l2_syn", np.nan),
           spl["dna_priv_l2_missense"], spl["dna_priv_l2_splice"]]
    ax[2].bar(np.arange(3), act, 0.6, color=[BASE, PD, CC])
    ax[2].set_xticks(np.arange(3)); ax[2].set_xticklabels(cls)
    ax[2].set_ylabel("DNA-private code $\\|z\\|$")
    ax[2].set_title(f"(c) DNA-private tracks DNA impact\n(splice-vs-missense AUROC {spl['splice_vs_mis_auroc_cv']:.2f})")

    fig.tight_layout()
    out = Path("figures"); out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_extra.pdf"); fig.savefig(out / "fig_extra.png", dpi=150)
    print("wrote figures/fig_extra.pdf")


if __name__ == "__main__":
    main()
