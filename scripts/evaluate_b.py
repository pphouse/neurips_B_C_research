#!/usr/bin/env python3
"""Evaluate Research B: retrieval, DMS/ClinVar prediction, interpretability."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder
from cdd.eval.probes import (
    ridge_spearman, logistic_auroc, cca_transform, retrieval_recall,
)
from cdd.utils.common import load_yaml, save_json


def load_model(run_dir, dev):
    ck = torch.load(Path(run_dir) / "crosscoder.pt", map_location=dev, weights_only=False)
    cfg = CrosscoderConfig(**ck["cfg"])
    m = SharedPrivateCrosscoder(cfg).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def codes(model, Xd, Xp, dev):
    with torch.no_grad():
        out = model(torch.tensor(Xd, device=dev), torch.tensor(Xp, device=dev))
        zs_d = out["zs_d"].cpu().numpy()
        zs_p = out["zs_p"].cpu().numpy()
        zp_d = out["zp_d"].cpu().numpy()
        zp_p = out["zp_p"].cpu().numpy()
    return zs_d, zs_p, zp_d, zp_p


def enrichment(z, labels, label_names):
    """For each latent, AUROC of latent activation predicting each binary annotation."""
    res = {}
    active = (z > 0)
    for name, y in zip(label_names, labels):
        y = np.asarray(y, float)
        keep = ~np.isnan(y)
        if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
            continue
        aucs = []
        for f in range(z.shape[1]):
            zf = z[keep, f]
            if active[keep, f].sum() < 5:
                aucs.append(np.nan); continue
            try:
                aucs.append(roc_auc_score(y[keep], zf))
            except Exception:
                aucs.append(np.nan)
        aucs = np.array(aucs)
        res[name] = dict(best_latent=int(np.nanargmax(np.abs(aucs - 0.5)) if np.isfinite(aucs).any() else -1),
                         best_auroc=float(np.nanmax(aucs)) if np.isfinite(aucs).any() else float("nan"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
                      pooling=cfg.get("pooling", "exact"),
                      norm_split_col=cfg.get("split_col", "split_position"))
    meta = pdd.meta
    model, _ = load_model(args.run_dir, dev)
    results = {"run_dir": args.run_dir, "dna_layer": cfg["dna_layer"], "prot_layer": cfg["prot_layer"]}

    for split_col in ["split_position", "split_domain"]:
        if split_col not in meta:
            continue
        sp = meta[split_col].to_numpy()
        tr, te = sp == "train", sp == "test"
        if te.sum() < 20:
            continue
        zs_d, zs_p, zp_d, zp_p = codes(model, pdd.dna, pdd.prot, dev)

        # ---- cross-modal retrieval on test (shared codes) ----
        ret_cc = retrieval_recall(zs_d[te], zs_p[te])
        # CCA baseline retrieval
        cca, _, Za_te, Zb_te = cca_transform(pdd.dna[tr], pdd.prot[tr], pdd.dna[te], pdd.prot[te],
                                             n_comp=cfg.get("k_shared", 32))
        ret_cca = retrieval_recall(Za_te, Zb_te)

        # ---- DMS prediction ----
        y = meta["dms_score"].to_numpy()
        keep = ~np.isnan(y)
        trk, tek = tr & keep, te & keep
        dms = {}
        dms["dna_only"] = ridge_spearman(pdd.dna[trk], y[trk], pdd.dna[tek], y[tek])[0]
        dms["prot_only"] = ridge_spearman(pdd.prot[trk], y[trk], pdd.prot[tek], y[tek])[0]
        concat = np.concatenate([pdd.dna, pdd.prot], 1)
        dms["concat"] = ridge_spearman(concat[trk], y[trk], concat[tek], y[tek])[0]
        dms["shared_code"] = ridge_spearman(np.concatenate([zs_d, zs_p], 1)[trk], y[trk],
                                            np.concatenate([zs_d, zs_p], 1)[tek], y[tek])[0]
        dms["shared_dna_code"] = ridge_spearman(zs_d[trk], y[trk], zs_d[tek], y[tek])[0]
        # CCA components probe
        _, Za_all_tr, _, _ = cca_transform(pdd.dna[trk], pdd.prot[trk], pdd.dna[tek], pdd.prot[tek],
                                           n_comp=cfg.get("k_shared", 32))
        cca2, Zc_tr, Zc_te, _ = cca_transform(pdd.dna[trk], pdd.prot[trk], pdd.dna[tek], pdd.prot[tek],
                                              n_comp=cfg.get("k_shared", 32))
        dms["cca"] = ridge_spearman(Zc_tr, y[trk], Zc_te, y[tek])[0]
        # external predictors (reference)
        ext = {}
        for col, sign in [("cadd", 1), ("phylop", 1), ("polyphen2", 1), ("sift", -1)]:
            v = pd.to_numeric(meta[col], errors="coerce").to_numpy()
            m2 = tek & ~np.isnan(v)
            if m2.sum() > 20:
                ext[col] = float(spearmanr(sign * v[m2], y[m2]).correlation)

        results[split_col] = dict(
            n_test=int(te.sum()),
            retrieval_crosscoder=ret_cc, retrieval_cca=ret_cca,
            dms=dms, dms_external=ext,
        )
        print(f"[{split_col}] retrieval CC R@1={ret_cc['R@1']:.3f} CCA R@1={ret_cca['R@1']:.3f} | "
              f"DMS shared={dms['shared_code']:.3f} dna={dms['dna_only']:.3f} "
              f"prot={dms['prot_only']:.3f} concat={dms['concat']:.3f} cca={dms['cca']:.3f}")

    # ---- ClinVar AUROC (all variants, position split) ----
    yc = meta["clinvar_bin"].to_numpy().astype(float)
    keepc = ~np.isnan(yc)
    sp = meta["split_position"].to_numpy()
    trc = (sp == "train") & keepc
    tec = (sp == "test") & keepc
    zs_d, zs_p, _, _ = codes(model, pdd.dna, pdd.prot, dev)
    clin = {}
    if tec.sum() > 10 and len(np.unique(yc[tec])) == 2:
        feat = np.concatenate([zs_d, zs_p], 1)
        clin["shared_code"] = logistic_auroc(feat[trc], yc[trc], feat[tec], yc[tec])[0]
        clin["dna_only"] = logistic_auroc(pdd.dna[trc], yc[trc], pdd.dna[tec], yc[tec])[0]
        clin["prot_only"] = logistic_auroc(pdd.prot[trc], yc[trc], pdd.prot[tec], yc[tec])[0]
        for col, sign in [("cadd", 1), ("phylop", 1)]:
            v = pd.to_numeric(meta[col], errors="coerce").to_numpy()
            m2 = tec & ~np.isnan(v)
            if m2.sum() > 10 and len(np.unique(yc[m2])) == 2:
                clin[col] = float(roc_auc_score(yc[m2], sign * v[m2]))
    results["clinvar_auroc"] = clin
    results["clinvar_n_test"] = int(tec.sum())

    # ---- interpretability: shared-code enrichment vs domain / LOF ----
    labels = [
        (meta["domain"] == "RING").astype(float).to_numpy(),
        (meta["domain"] == "BRCT").astype(float).to_numpy(),
        (meta["func_class"] == "LOF").astype(float).to_numpy(),
    ]
    names = ["domain_RING", "domain_BRCT", "func_LOF"]
    results["enrichment_shared_dna"] = enrichment(zs_d, labels, names)
    _, _, zp_d, zp_p = codes(model, pdd.dna, pdd.prot, dev)
    results["enrichment_private_dna"] = enrichment(zp_d, labels, names)
    results["enrichment_private_prot"] = enrichment(zp_p, labels, names)

    save_json(results, Path(args.run_dir) / "eval_b.json")
    print("Saved", Path(args.run_dir) / "eval_b.json")
    print(json.dumps({k: results.get(k) for k in ["clinvar_auroc"]}, indent=2))


if __name__ == "__main__":
    main()
