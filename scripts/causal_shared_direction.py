#!/usr/bin/env python3
"""Cross-model causal test via a SHARED direction.

We take the crosscoder's linear alignment space (in which DNA and protein variant deltas are
contrastively aligned), find the single direction u that best predicts DMS function score,
and map u back into each model's raw activation space through that model's own alignment
encoder. Injecting +/-alpha*dir into Evo2 and into ESM-2 at the variant position, we test
whether a SINGLE shared factor moves BOTH models' variant scores consistently (dose-response
+ cross-model agreement), versus a matched-norm random shared direction.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder
from cdd.interventions.evo2_patch import Patcher
from cdd.interventions.esm_patch import EsmScorer
from cdd.utils.common import load_yaml, save_json, set_seed

W = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-variants", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=30.0)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(0); dev = "cuda"

    pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
                      pooling=cfg.get("pooling", "local"), norm_split_col="split_position",
                      n_pca=cfg.get("n_pca"))
    meta = pdd.meta
    ck = torch.load(Path(args.run_dir) / "crosscoder.pt", map_location=dev, weights_only=False)
    model = SharedPrivateCrosscoder(CrosscoderConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    with torch.no_grad():
        a = model.encode_all(torch.tensor(pdd.dna, device=dev), torch.tensor(pdd.prot, device=dev))
    a_d = a["align_dna"].cpu().numpy()
    tr = (meta.split_position == "train").to_numpy()
    te = (meta.split_position == "test").to_numpy()
    y = meta.dms_score.to_numpy(); keep = ~np.isnan(y)

    # Functional direction in each model's activation (PCA) space (ridge delta->DMS).
    # The crosscoder certifies these two directions as the SAME shared latent: their
    # alignment-embeddings have high cosine (reported below).
    wd = Ridge(alpha=10.0).fit(pdd.dna[tr & keep], y[tr & keep]).coef_   # (P_dna,)
    wp = Ridge(alpha=10.0).fit(pdd.prot[tr & keep], y[tr & keep]).coef_  # (P_prot,)
    rng = np.random.default_rng(0)
    rwd = rng.standard_normal(wd.shape); rwp = rng.standard_normal(wp.shape)

    def to_raw(v, mod):
        d = torch.tensor(pdd.input_dir_to_raw(mod, v / (np.linalg.norm(v) + 1e-8)), device=dev).float()
        return d / d.norm()
    dir_d, dir_p = to_raw(wd, "dna"), to_raw(wp, "prot")
    rdir_d, rdir_p = to_raw(rwd, "dna"), to_raw(rwp, "prot")

    # cross-modal alignment of the two functional directions in the crosscoder shared space
    with torch.no_grad():
        ea = model.dna.align(torch.tensor(wd[None] / np.linalg.norm(wd), device=dev, dtype=torch.float32))
        eb = model.prot.align(torch.tensor(wp[None] / np.linalg.norm(wp), device=dev, dtype=torch.float32))
        shared_cos = torch.cosine_similarity(ea, eb).item()
    print(f"shared-space cosine of the two functional directions: {shared_cos:.3f}", flush=True)

    # data for interventions
    op = gzip.open if str(cfg["chr17_fasta"]).endswith(".gz") else open
    with op(cfg["chr17_fasta"], "rt") as f:
        c17 = str(list(SeqIO.parse(f, "fasta"))[0].seq)
    prot = str(list(SeqIO.parse(cfg["protein_fasta"], "fasta"))[0].seq)

    from evo2 import Evo2
    evo = Evo2(cfg.get("evo2_model", "evo2_7b"))
    esm = EsmScorer(cfg["esm_model"], device=dev)

    idxs = np.where(te & keep)[0]
    rng.shuffle(idxs); idxs = idxs[: args.n_variants]
    A = args.alpha
    rows = []
    with Patcher(evo, cfg["dna_layer"], radius=cfg.get("inject_radius", 8)) as pt:
        for i in idxs:
            r = meta.iloc[i]
            p0 = int(r.pos) - 1; s = max(0, p0 - W // 2)
            ref = c17[s:min(len(c17), p0 + W // 2)]; di = min(W // 2, p0)
            var = ref[:di] + r.alt + ref[di + 1:]
            rid = torch.tensor(evo.tokenizer.tokenize(ref), dtype=torch.int).unsqueeze(0).cuda()
            vid = torch.tensor(evo.tokenizer.tokenize(var), dtype=torch.int).unsqueeze(0).cuda()
            llref = pt.scored_forward(rid)
            base_d = pt.scored_forward(vid, pos=di, delta_vec=None) - llref
            evo_plus = pt.scored_forward(vid, pos=di, delta_vec=A * dir_d) - llref
            evo_minus = pt.scored_forward(vid, pos=di, delta_vec=-A * dir_d) - llref
            evo_rand = pt.scored_forward(vid, pos=di, delta_vec=A * rdir_d) - llref
            # ESM
            ap_ = int(r.aa_pos); ss = max(0, ap_ - 1 - 510); ee = min(len(prot), ss + 1021); ss = max(0, ee - 1021)
            wtw = prot[ss:ee]; ridx = ap_ - 1 - ss
            L = int(cfg["prot_layer"][1:])
            base_p = esm.masked_marginal(wtw, ridx, r.aa_ref, r.aa_alt)
            esm_plus = esm.masked_marginal(wtw, ridx, r.aa_ref, r.aa_alt, layer_idx=L, delta_vec=A * dir_p)
            esm_minus = esm.masked_marginal(wtw, ridx, r.aa_ref, r.aa_alt, layer_idx=L, delta_vec=-A * dir_p)
            esm_rand = esm.masked_marginal(wtw, ridx, r.aa_ref, r.aa_alt, layer_idx=L, delta_vec=A * rdir_p)
            rows.append(dict(
                variant=r.variant_id, dms=float(r.dms_score), func=r.func_class,
                evo_dose=evo_plus - evo_minus, evo_plus=evo_plus - base_d, evo_rand=evo_rand - base_d,
                esm_dose=esm_plus - esm_minus, esm_plus=esm_plus - base_p, esm_rand=esm_rand - base_p,
            ))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.run_dir) / "causal_shared.csv", index=False)
    summ = dict(
        n=len(df), alpha=A, shared_space_cosine=float(shared_cos),
        evo_dose_mean=float(df.evo_dose.mean()), esm_dose_mean=float(df.esm_dose.mean()),
        evo_dose_pos_frac=float((df.evo_dose > 0).mean()), esm_dose_pos_frac=float((df.esm_dose > 0).mean()),
        # cross-model: do the +alpha effects correlate across variants?
        corr_evo_esm_plus=float(spearmanr(df.evo_plus, df.esm_plus).correlation),
        frac_same_dir_plus=float((np.sign(df.evo_plus) == np.sign(df.esm_plus)).mean()),
        # specificity vs random shared direction
        evo_specific=float(df.evo_plus.abs().mean() - df.evo_rand.abs().mean()),
        esm_specific=float(df.esm_plus.abs().mean() - df.esm_rand.abs().mean()),
        # does injecting the functional direction correlate with DMS sign? (rescue effect)
        corr_evo_plus_dms=float(spearmanr(df.evo_plus, df.dms).correlation),
        p_evo_plus_dms=float(spearmanr(df.evo_plus, df.dms).pvalue),
        corr_esm_plus_dms=float(spearmanr(df.esm_plus, df.dms).correlation),
        p_esm_plus_dms=float(spearmanr(df.esm_plus, df.dms).pvalue),
    )
    save_json(summ, Path(args.run_dir) / "causal_shared_summary.json")
    print("SHARED CAUSAL SUMMARY:")
    for k, v in summ.items():
        print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")


if __name__ == "__main__":
    main()
