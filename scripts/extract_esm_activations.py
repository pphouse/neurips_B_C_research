#!/usr/bin/env python3
"""Extract ESM-2 WT/mut activations at the variant residue for BRCA1 missense variants.

BRCA1 (1863 aa) exceeds ESM-2's context, so we take a variant-centred window of up to
1021 residues. WT and mutant windows differ by exactly one residue (the substitution).
We store the pooled hidden state at the variant residue (exact + local mean) for each
selected transformer layer; all layers come from a single forward via output_hidden_states.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from cdd.activations.pooling import pool_exact, pool_local_mean, to_np
from cdd.utils.common import load_yaml, save_json, set_seed

MAXLEN = 1021  # ESM-2 usable window (1024 minus CLS/EOS/margin)


def load_protein(fasta: str) -> str:
    return str(list(SeqIO.parse(fasta, "fasta"))[0].seq)


def window(protein: str, aa_pos: int, aa_alt: str, radius: int = 510):
    """Return (wt_window, mut_window, idx_in_window). aa_pos is 1-based."""
    p = aa_pos - 1
    start = max(0, p - radius)
    end = min(len(protein), start + MAXLEN)
    start = max(0, end - MAXLEN)
    wt = protein[start:end]
    idx = p - start
    mut = wt[:idx] + aa_alt + wt[idx + 1 :]
    return wt, mut, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 0))

    outdir = Path(cfg["out_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    layers = cfg["layers"]  # transformer layer indices, e.g. [6, 18, 30, 33]
    radius = cfg.get("local_radius", 8)

    df = pd.read_parquet(cfg["table"])
    df = df[df["paired"]].reset_index(drop=True)  # missense only
    if cfg.get("limit"):
        df = df.iloc[: int(cfg["limit"])].reset_index(drop=True)
    n = len(df)
    protein = load_protein(cfg["protein_fasta"])
    print(f"Extracting ESM-2 activations for {n} missense variants, layers={layers}")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(cfg["esm_model"])
    model = AutoModel.from_pretrained(cfg["esm_model"], torch_dtype=torch.float32).cuda().eval()
    D = model.config.hidden_size

    store = {f"L{l}_{k}": np.zeros((n, D), np.float32)
             for l in layers for k in ("wt_exact", "mut_exact", "wt_local", "mut_local")}
    ok = np.zeros(n, bool)
    ckpt = outdir / "esm_store.npz"

    def run(seq: str):
        enc = tok(seq, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        # hidden_states[l]: (1, L+2, D); index 0 is CLS -> residue r is at position r+1
        return out.hidden_states

    t0 = time.time()
    for i in range(n):
        r = df.iloc[i]
        wt, mut, idx = window(protein, int(r.aa_pos), r.aa_alt, cfg.get("radius", 510))
        assert wt[idx] == r.aa_ref, f"aa_ref mismatch {wt[idx]}!={r.aa_ref} @ {r.aa_pos}"
        hs_wt = run(wt)
        hs_mut = run(mut)
        tok_idx = idx + 1  # +1 for CLS
        for l in layers:
            aw = hs_wt[l][0]
            am = hs_mut[l][0]
            store[f"L{l}_wt_exact"][i] = to_np(pool_exact(aw, tok_idx))
            store[f"L{l}_mut_exact"][i] = to_np(pool_exact(am, tok_idx))
            store[f"L{l}_wt_local"][i] = to_np(pool_local_mean(aw, tok_idx, radius))
            store[f"L{l}_mut_local"][i] = to_np(pool_local_mean(am, tok_idx, radius))
        ok[i] = True
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  {i+1}/{n} ({(time.time()-t0)/60:.1f}min)", flush=True)
            np.savez(ckpt, ok=ok, **store)
            df.to_parquet(outdir / "index.parquet")

    np.savez(ckpt, ok=ok, **store)
    df.to_parquet(outdir / "index.parquet")
    save_json({"n": n, "n_ok": int(ok.sum()), "layers": layers, "esm_model": cfg["esm_model"],
               "hidden": D, "radius": radius}, outdir / "esm_meta.json")
    print(f"Done. {int(ok.sum())}/{n} -> {ckpt}")


if __name__ == "__main__":
    main()
