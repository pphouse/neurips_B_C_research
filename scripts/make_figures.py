#!/usr/bin/env python3
"""Generate the main results figure from the real result JSONs."""
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
    pos = L("outputs/b_mvp/seeds_summary.json")
    dom = L("outputs/b_domain/seeds_summary.json")
    ab = L("outputs/b_mvp/ablations.json")
    coll = {k: v["auroc"] if isinstance(v, dict) else v for k, v in ab["probe"].items()}
    cc, cb = pos["crosscoder"], pos["baselines"]
    dc, db = dom["crosscoder"], dom["baselines"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    # (a) retrieval ablation (residue-disjoint): CCA vs deepCCA vs sparse vs alignment head
    labels = ["linear\nCCA", "deep\nCCA", "sparse\ncode", "align\nhead"]
    r1 = [cb["retrieval_cca"]["R@1"], ab["deep_cca"]["R1"], ab["sparse_shared"]["R1"], cc["R1"][0]]
    r10 = [cb["retrieval_cca"]["R@10"], ab["deep_cca"]["R10"], ab["sparse_shared"]["R10"], cc["R10"][0]]
    x = np.arange(4); w = 0.38
    ax[0].bar(x - w/2, r1, w, label="R@1", color=CC)
    ax[0].bar(x + w/2, r10, w, label="R@10", color=PD)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("Recall")
    ax[0].set_title("(a) Cross-modal retrieval (residue-disjoint)"); ax[0].legend(fontsize=8)

    # (b) DMS by method, position + domain
    methods = ["Evo2\nΔ", "ESM2\nΔ", "CCA", "concat", "shared\n(ours)"]
    posv = [cb["dms_dna"], cb["dms_prot"], cb["dms_cca"], cb["dms_concat"], cc["dms"][0]]
    domv = [db["dms_dna"], db["dms_prot"], db["dms_cca"], db["dms_concat"], dc["dms"][0]]
    x = np.arange(5)
    ax[1].bar(x - w/2, posv, w, label="residue-disjoint", color=CC)
    ax[1].bar(x + w/2, domv, w, label="domain-disjoint", color=PD)
    cols = [BASE]*4 + [SH]
    for i, b in enumerate(ax[1].patches):
        pass
    ax[1].set_xticks(x); ax[1].set_xticklabels(methods); ax[1].set_ylabel("DMS Spearman ρ")
    ax[1].set_title("(b) Function-score prediction"); ax[1].legend(fontsize=8)

    # (c) probing AUROC with CIs (shared vs protein-private)
    P = ab["probe"]
    labels = ["LOF vs FUNC", "RING vs BRCT"]
    shared = [P["LOF/shared"]["auroc"], P["RING/shared"]["auroc"]]
    shared_e = [[shared[0]-P["LOF/shared"]["ci"][0], shared[1]-P["RING/shared"]["ci"][0]],
                [P["LOF/shared"]["ci"][1]-shared[0], P["RING/shared"]["ci"][1]-shared[1]]]
    pprot = [P["LOF/priv_prot"]["auroc"], P["RING/priv_prot"]["auroc"]]
    pprot_e = [[pprot[0]-P["LOF/priv_prot"]["ci"][0], pprot[1]-P["RING/priv_prot"]["ci"][0]],
               [P["LOF/priv_prot"]["ci"][1]-pprot[0], P["RING/priv_prot"]["ci"][1]-pprot[1]]]
    x = np.arange(2); w3 = 0.34
    ax[2].bar(x - w3/2, shared, w3, yerr=shared_e, capsize=3, label="shared", color=SH)
    ax[2].bar(x + w3/2, pprot, w3, yerr=pprot_e, capsize=3, label="protein-private", color=PP)
    ax[2].axhline(0.5, ls=":", c="k", lw=1)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].set_ylabel("AUROC")
    ax[2].set_ylim(0.5, 0.95); ax[2].set_title("(c) Biological probing of codes")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = Path("figures"); out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_results.pdf"); fig.savefig(out / "fig_results.png", dpi=150)
    print("wrote figures/fig_results.pdf")


if __name__ == "__main__":
    main()
