#!/usr/bin/env python3
"""Gene-disjoint generalization: train the crosscoder on a set of genes and evaluate on
held-out genes (cross-modal retrieval within each held-out gene + gene-disjoint ClinVar
pathogenicity prediction), vs. linear CCA and deep-CCA. This directly tests whether the
learned cross-modal alignment transfers to unseen genes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cdd.crosscoder.model import (CrosscoderConfig, SharedPrivateCrosscoder,
                                  crosscoder_loss, info_nce)
from cdd.eval.probes import retrieval_recall, cca_transform
from cdd.utils.common import set_seed, save_json
import pandas as pd


def load_mg(evo_dir, esm_dir, n_pca=128):
    zd = np.load(Path(evo_dir) / "evo2_store.npz"); zp = np.load(Path(esm_dir) / "esm_store.npz")
    idx = pd.read_parquet(Path(evo_dir) / "index.parquet")
    ok = zd["ok"] & zp["ok"]
    dna = (zd["mut"] - zd["wt"])[ok].astype(np.float32)
    prot = (zp["mut"] - zp["wt"])[ok].astype(np.float32)
    meta = idx[ok].reset_index(drop=True)
    tr = (meta.split_gene == "train").to_numpy()
    ds = dna[tr].std(0) + 1e-6; ps = prot[tr].std(0) + 1e-6
    dna /= ds; prot /= ps
    pd_ = PCA(n_pca, whiten=True, svd_solver="full").fit(dna[tr])
    pp_ = PCA(n_pca, whiten=True, svd_solver="full").fit(prot[tr])
    return pd_.transform(dna).astype(np.float32), pp_.transform(prot).astype(np.float32), meta


def train_cc(Xd, Xp, tr, seed, dev, steps=4000):
    set_seed(seed)
    cfg = CrosscoderConfig(d_dna=Xd.shape[1], d_prot=Xp.shape[1], k_shared=32, k_private=96,
                           topk_shared=24, topk_private=24, d_align=64)
    mo = SharedPrivateCrosscoder(cfg).to(dev); opt = torch.optim.Adam(mo.parameters(), 1e-3)
    Xdt, Xpt = Xd[tr], Xp[tr]; n = Xdt.shape[0]
    for s in range(steps):
        idx = torch.randint(0, n, (256,), device=dev); out = mo(Xdt[idx], Xpt[idx])
        ramp = min(1., max(0., (s - 800) / 1200.))
        w = dict(rec=1., align=0.05 * ramp, contrast=2.0, orth=0.1 * ramp, temp=0.1)
        loss, _ = crosscoder_loss(out, Xdt[idx], Xpt[idx], w); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mo.parameters(), 1.); opt.step()
    mo.eval(); return mo


def per_gene_retrieval(za, zp, genes, test_genes):
    """Retrieve within each held-out gene separately (removes gene-identity shortcut)."""
    r1, r10, ns = [], [], []
    for g in test_genes:
        m = genes == g
        if m.sum() < 5:
            continue
        rr = retrieval_recall(za[m], zp[m])
        r1.append(rr["R@1"] * m.sum()); r10.append(rr["R@10"] * m.sum()); ns.append(m.sum())
    N = sum(ns)
    return dict(R1=float(sum(r1) / N), R10=float(sum(r10) / N), n=int(N), n_genes=len(ns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo-dir", default="outputs/act/mg_evo2")
    ap.add_argument("--esm-dir", default="outputs/act/mg_esm")
    ap.add_argument("--out", default="outputs/multigene/eval_mg.json")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xd, Xp, meta = load_mg(args.evo_dir, args.esm_dir)
    tr = (meta.split_gene == "train").to_numpy(); te = ~tr
    genes = meta.gene.to_numpy(); test_genes = sorted(meta[te].gene.unique())
    y = meta.clinvar_bin.to_numpy().astype(int)
    Xdt = torch.tensor(Xd, device=dev); Xpt = torch.tensor(Xp, device=dev)
    print(f"MG: N={len(meta)} train={tr.sum()} test={te.sum()} test_genes={test_genes}")

    # ---- baselines (deterministic): CCA + deep-CCA trained on train genes ----
    def deepcca(seed=0, steps=4000):
        set_seed(seed); Ed = torch.nn.Linear(Xd.shape[1], 64).to(dev); Ep = torch.nn.Linear(Xp.shape[1], 64).to(dev)
        opt = torch.optim.Adam(list(Ed.parameters()) + list(Ep.parameters()), 1e-3)
        A, B = Xdt[tr], Xpt[tr]; nn = A.shape[0]
        for s in range(steps):
            i = torch.randint(0, nn, (256,), device=dev)
            loss = info_nce(Ed(A[i]), Ep(B[i]), 0.1); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return Ed(Xdt).cpu().numpy(), Ep(Xpt).cpu().numpy()
    res = {"test_genes": test_genes, "n_test": int(te.sum())}
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=32, max_iter=500, tol=1e-4).fit(Xd[tr], Xp[tr])
    ad_cca_d, ap_cca_p = cca.transform(Xd, Xp)  # (N,32) each, full array
    res["retrieval_cca"] = per_gene_retrieval(ad_cca_d[te], ap_cca_p[te], genes[te], test_genes)
    dd, dp = deepcca()
    res["retrieval_deepcca"] = per_gene_retrieval(dd[te], dp[te], genes[te], test_genes)

    # ---- crosscoder over seeds ----
    r1s, r10s, aucs, auc_dna, auc_prot, auc_cca = [], [], [], [], [], []
    for seed in range(args.seeds):
        mo = train_cc(Xdt, Xpt, tr, seed, dev)
        with torch.no_grad():
            a = mo.encode_all(Xdt, Xpt)
        ad = a["align_dna"].cpu().numpy(); ap = a["align_prot"].cpu().numpy()
        rr = per_gene_retrieval(ad[te], ap[te], genes[te], test_genes)
        r1s.append(rr["R1"]); r10s.append(rr["R10"])
        # gene-disjoint ClinVar AUROC: train logistic on train-gene shared code, test on held-out genes
        sh = np.concatenate([ad, ap], 1)
        def auc(feat):
            sc = StandardScaler().fit(feat[tr]); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feat[tr]), y[tr])
            return roc_auc_score(y[te], clf.predict_proba(sc.transform(feat[te]))[:, 1])
        aucs.append(auc(sh))
        if seed == 0:
            auc_dna.append(auc(Xd)); auc_prot.append(auc(Xp))
            auc_cca.append(auc(np.concatenate([ad_cca_d, ap_cca_p], 1)))
    res["retrieval_crosscoder"] = dict(R1=[round(float(np.mean(r1s)),4), round(float(np.std(r1s)),4)],
                                       R10=round(float(np.mean(r10s)),4), n=rr["n"], n_genes=rr["n_genes"])
    res["clinvar_auroc"] = dict(shared=[round(float(np.mean(aucs)),4), round(float(np.std(aucs)),4)],
                                dna=round(float(auc_dna[0]),4), prot=round(float(auc_prot[0]),4),
                                cca=round(float(auc_cca[0]),4))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); save_json(res, args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
