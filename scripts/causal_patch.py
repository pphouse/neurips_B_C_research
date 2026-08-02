#!/usr/bin/env python3
"""Principled causal test: activation PATCHING along the shared functional axis. For a mutant,
at the variant position, we remove the variant-induced change *along* the functional direction
(replace the mutant's projection with the wild-type's), and measure whether the Evo2 / ESM-2
variant score moves toward wild-type in proportion to the variant's true effect. Compares to
patching along a matched-norm random direction (specificity)."""
import argparse, gzip, glob, json
from pathlib import Path
import numpy as np, torch
from Bio import SeqIO
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge

from cdd.crosscoder.data import load_paired
from cdd.interventions.evo2_patch import Patcher
from cdd.interventions.esm_patch import EsmScorer
from cdd.utils.common import set_seed
W = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--layer", default="blocks.24.mlp.l3")
    ap.add_argument("--config", default="configs/experiments/causal.yaml")
    args = ap.parse_args(); set_seed(0); dev = "cuda"
    from cdd.utils.common import load_yaml; cfg = load_yaml(args.config)
    pdd = load_paired("outputs/act/evo2", "outputs/act/esm", args.layer, "L33", pooling="local", n_pca=128)
    meta = pdd.meta
    tr = (meta.split_position == "train").to_numpy(); te = (meta.split_position == "test").to_numpy()
    y = meta.dms_score.to_numpy(); keep = ~np.isnan(y)
    # functional direction in RAW activation space (ridge on raw local delta -> DMS)
    ev = np.load("outputs/act/evo2/evo2_store.npz"); idxe = __import__("pandas").read_parquet("outputs/act/evo2/index.parquet")
    oke = ev["ok"]; L = args.layer
    wt_all = ev[f"{L}_wt_local"][oke]; mut_all = ev[f"{L}_mut_local"][oke]
    me = idxe[oke].reset_index(drop=True)
    # map raw rows to pdd (paired missense) by variant_id
    vpos = {v: i for i, v in enumerate(me.variant_id)}
    rows = [vpos[v] for v in meta.variant_id]
    wt_raw = wt_all[rows]; mut_raw = mut_all[rows]; draw = (mut_raw - wt_raw)
    dfun = Ridge(alpha=10).fit(draw[tr & keep], y[tr & keep]).coef_
    dfun = dfun / np.linalg.norm(dfun)
    rng = np.random.default_rng(0); drand = rng.standard_normal(dfun.shape); drand /= np.linalg.norm(drand)
    dfun_t = torch.tensor(dfun, device=dev, dtype=torch.float32); drand_t = torch.tensor(drand, device=dev, dtype=torch.float32)

    op = gzip.open if str(cfg["chr17_fasta"]).endswith(".gz") else open
    with op(cfg["chr17_fasta"], "rt") as f: c17 = str(list(SeqIO.parse(f, "fasta"))[0].seq)
    prot = str(list(SeqIO.parse(cfg["protein_fasta"], "fasta"))[0].seq)
    from evo2 import Evo2; evo = Evo2("evo2_7b"); esm = EsmScorer(cfg["esm_model"], dev)

    idxs = np.where(te & keep)[0]; rng.shuffle(idxs); idxs = idxs[: args.n]
    res = []
    with Patcher(evo, args.layer, radius=cfg.get("inject_radius", 8)) as pt:
        for i in idxs:
            r = meta.iloc[i]; p0 = int(r.pos) - 1; s = max(0, p0 - W // 2)
            ref = c17[s:min(len(c17), p0 + W // 2)]; di = min(W // 2, p0)
            var = ref[:di] + r.alt + ref[di + 1:]
            rid = torch.tensor(evo.tokenizer.tokenize(ref), dtype=torch.int).unsqueeze(0).cuda()
            vid = torch.tensor(evo.tokenizer.tokenize(var), dtype=torch.int).unsqueeze(0).cuda()
            llref = pt.scored_forward(rid)
            base = pt.scored_forward(vid, pos=di, delta_vec=None) - llref
            # remove mutant's change along functional axis: patch = -(proj_mut - proj_wt) * d
            proj = float((mut_raw[i] - wt_raw[i]) @ dfun)
            patch_fun = (-proj) * dfun_t
            patch_rnd = (-float((mut_raw[i] - wt_raw[i]) @ drand)) * drand_t
            abl_fun = pt.scored_forward(vid, pos=di, delta_vec=patch_fun) - llref
            abl_rnd = pt.scored_forward(vid, pos=di, delta_vec=patch_rnd) - llref
            # ESM masked-marginal
            ap_ = int(r.aa_pos); ss = max(0, ap_ - 1 - 510); ee = min(len(prot), ss + 1021); ss = max(0, ee - 1021)
            wtw = prot[ss:ee]; ridx = ap_ - 1 - ss
            res.append(dict(v=r.variant_id, dms=float(r.dms_score), func=r.func_class,
                            base=base, abl_fun=abl_fun, abl_rnd=abl_rnd,
                            move_fun=abl_fun - base, move_rnd=abl_rnd - base))
    import pandas as pd; df = pd.DataFrame(res)
    # ablating the functional axis should move the mutant score toward WT (toward 0),
    # i.e. |abl_fun| < |base|, more so for damaging (low DMS / LOF) variants.
    toward_wt = (df.base.abs() - df.abl_fun.abs())          # >0 means moved toward WT
    toward_wt_rnd = (df.base.abs() - df.abl_rnd.abs())
    summ = dict(n=len(df),
        mean_toward_wt_fun=float(toward_wt.mean()), mean_toward_wt_rnd=float(toward_wt_rnd.mean()),
        frac_toward_wt_fun=float((toward_wt > 0).mean()),
        specificity=float(toward_wt.mean() - toward_wt_rnd.mean()),
        corr_move_dms=float(spearmanr(df.move_fun, df.dms).correlation),
        p_move_dms=float(spearmanr(df.move_fun, df.dms).pvalue))
    df.to_csv("outputs/b_mvp/causal_patch.csv", index=False)
    json.dump(summ, open("outputs/b_mvp/causal_patch.json", "w"), indent=2)
    print("PATCH CAUSAL:", json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
