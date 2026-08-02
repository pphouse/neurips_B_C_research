#!/usr/bin/env python3
"""Cross-model causal test: ablate a shared latent's decoder direction in BOTH Evo2 and
ESM-2, and measure whether each model's variant score moves toward wild-type together.

For a shared latent f we take its DNA decoder column (Evo2 mlp.l3 space, un-normalised by
the training std) and its protein decoder column (ESM hidden space). For variants that
strongly activate f we subtract that latent's reconstructed contribution at the variant
position and re-score. A matched-norm random direction is the control.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder
from cdd.interventions.evo2_patch import Patcher
from cdd.interventions.esm_patch import EsmScorer
from cdd.utils.common import load_yaml, save_json, set_seed

WINDOW = 8192


def load_chr17(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return str(list(SeqIO.parse(f, "fasta"))[0].seq)


def parse_seq(seq_chr17, pos, ref, alt):
    p = pos - 1
    s = max(0, p - WINDOW // 2); e = min(len(seq_chr17), p + WINDOW // 2)
    ref_seq = seq_chr17[s:e]; idx = min(WINDOW // 2, p)
    var_seq = ref_seq[:idx] + alt + ref_seq[idx + 1:]
    return ref_seq, var_seq, idx


def prot_window(protein, aa_pos, aa_alt, radius=510, maxlen=1021):
    p = aa_pos - 1
    s = max(0, p - radius); e = min(len(protein), s + maxlen); s = max(0, e - maxlen)
    wt = protein[s:e]; idx = p - s
    return wt, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-latents", type=int, default=4)
    ap.add_argument("--n-variants", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(0)
    dev = "cuda"

    pdd = load_paired(cfg["evo2_dir"], cfg["esm_dir"], cfg["dna_layer"], cfg["prot_layer"],
                      pooling=cfg.get("pooling", "exact"), norm_split_col="split_position")
    meta = pdd.meta
    ck = torch.load(Path(args.run_dir) / "crosscoder.pt", map_location=dev, weights_only=False)
    model = SharedPrivateCrosscoder(CrosscoderConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    dna_std = torch.tensor(ck["dna_std"], device=dev)
    prot_std = torch.tensor(ck["prot_std"], device=dev)

    with torch.no_grad():
        out = model(torch.tensor(pdd.dna, device=dev), torch.tensor(pdd.prot, device=dev))
        zs_d = out["zs_d"].cpu().numpy()

    # rank shared latents by |Spearman correlation of activation with DMS function score|
    from scipy.stats import spearmanr
    dms = meta["dms_score"].to_numpy()
    keepd = ~np.isnan(dms)
    scores = np.zeros(zs_d.shape[1])
    for f in range(zs_d.shape[1]):
        if (zs_d[keepd, f] > 0).sum() < 15:
            continue
        c = spearmanr(zs_d[keepd, f], dms[keepd]).correlation
        scores[f] = 0.0 if c != c else c
    top = np.argsort(-np.abs(scores))[: args.n_latents]
    print("Top shared latents (by |Spearman with DMS|):",
          [(int(f), round(float(scores[f]), 3)) for f in top])

    from evo2 import Evo2
    evo = Evo2(cfg.get("evo2_model", "evo2_7b"))
    esm = EsmScorer(cfg["esm_model"], device=dev)
    seq_chr17 = load_chr17(cfg["chr17_fasta"])
    protein = str(list(SeqIO.parse(cfg["protein_fasta"], "fasta"))[0].seq)

    # decoder columns (normalized space) -> raw space directions
    dec_d = model.dna.dec_shared.weight.detach()   # (D_dna, k_shared)
    dec_p = model.prot.dec_shared.weight.detach()   # (D_prot, k_shared)

    rng = np.random.default_rng(0)
    results = []
    test_mask = meta["split_position"].to_numpy() == "test"
    for f in top:
        dir_d_raw = (dec_d[:, f] * dna_std)   # raw Evo2 direction
        dir_p_raw = (dec_p[:, f] * prot_std)
        # variants (test) most activating latent f
        cand = np.where(test_mask & (zs_d[:, f] > 0))[0]
        cand = cand[np.argsort(-zs_d[cand, f])][: args.n_variants]
        for i in cand:
            r = meta.iloc[i]
            act = float(zs_d[i, f])
            # Evo2: cache ref LL once, then score var under base/ablation/control patches
            ref_seq, var_seq, idx = parse_seq(seq_chr17, int(r.pos), r.ref, r.alt)
            ref_ids = torch.tensor(evo.tokenizer.tokenize(ref_seq), dtype=torch.int).unsqueeze(0).cuda()
            var_ids = torch.tensor(evo.tokenizer.tokenize(var_seq), dtype=torch.int).unsqueeze(0).cuda()
            patch = (-args.alpha * act * dir_d_raw)
            rnd = torch.tensor(rng.standard_normal(dir_d_raw.shape[0]), device=dev, dtype=dir_d_raw.dtype)
            rnd = rnd / rnd.norm() * patch.norm()
            with Patcher(evo, cfg["dna_layer"]) as p:
                ll_ref = p.scored_forward(ref_ids)
                base_d = p.scored_forward(var_ids, pos=idx, delta_vec=None) - ll_ref
                abl_d = p.scored_forward(var_ids, pos=idx, delta_vec=patch) - ll_ref
                ctl_d = p.scored_forward(var_ids, pos=idx, delta_vec=rnd) - ll_ref
            # ESM ablation
            wt_win, ridx = prot_window(protein, int(r.aa_pos), r.aa_alt)
            L = int(cfg["prot_layer"][1:])
            base_p = esm.masked_marginal(wt_win, ridx, r.aa_ref, r.aa_alt)
            patch_p = (-args.alpha * act * dir_p_raw)
            abl_p = esm.masked_marginal(wt_win, ridx, r.aa_ref, r.aa_alt, layer_idx=L, delta_vec=patch_p)
            results.append(dict(latent=int(f), variant=r.variant_id, act=act,
                                func_class=r.func_class, dms=float(r.dms_score) if not pd.isna(r.dms_score) else None,
                                evo_base=base_d, evo_abl=abl_d, evo_ctl=ctl_d,
                                evo_delta=abl_d - base_d, evo_ctl_delta=ctl_d - base_d,
                                esm_base=base_p, esm_abl=abl_p, esm_delta=abl_p - base_p))
        print(f"latent {f}: done {len(cand)} variants", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(Path(args.run_dir) / "causal.csv", index=False)
    # summary: do Evo2 and ESM scores move together under shared-latent ablation?
    summ = dict(
        n=len(df),
        evo_mean_abl_delta=float(df.evo_delta.mean()),
        evo_mean_ctl_delta=float(df.evo_ctl_delta.mean()),
        esm_mean_abl_delta=float(df.esm_delta.mean()),
        corr_evo_esm_delta=float(spearmanr(df.evo_delta, df.esm_delta).correlation),
        frac_same_direction=float(np.mean(np.sign(df.evo_delta) == np.sign(df.esm_delta))),
        abl_vs_ctl_evo_effect=float(df.evo_delta.abs().mean() - df.evo_ctl_delta.abs().mean()),
    )
    save_json(summ, Path(args.run_dir) / "causal_summary.json")
    print("CAUSAL SUMMARY:", summ)


if __name__ == "__main__":
    main()
