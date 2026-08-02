#!/usr/bin/env python3
"""Extract Evo 2 WT/mut activations at the variant position for BRCA1 variants.

Replicates the BRCA1 zero-shot notebook's window construction (8192 bp, SNV-centered,
+ strand) so that Evo 2 sees exactly the sequences it was validated on. For each
selected layer we store the pooled activation at the SNV token for both the reference
and the variant window (exact position and local mean). Deltas are formed downstream.
"""
import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from cdd.activations.pooling import pool_exact, pool_local_mean, to_np
from cdd.utils.common import load_yaml, save_json, set_seed

WINDOW = 8192


def load_chr17(path: str) -> str:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        rec = list(SeqIO.parse(f, "fasta"))[0]
    return str(rec.seq)


def parse_sequences(seq_chr17: str, pos: int, ref: str, alt: str):
    """Verbatim window logic from the Evo2 BRCA1 notebook (+ strand, SNV-centered)."""
    p = pos - 1  # 0-based
    ref_start = max(0, p - WINDOW // 2)
    ref_end = min(len(seq_chr17), p + WINDOW // 2)
    ref_seq = seq_chr17[ref_start:ref_end]
    snv_idx = min(WINDOW // 2, p)
    var_seq = ref_seq[:snv_idx] + alt + ref_seq[snv_idx + 1 :]
    assert len(var_seq) == len(ref_seq)
    assert ref_seq[snv_idx] == ref, f"ref mismatch at pos {pos}: {ref_seq[snv_idx]} != {ref}"
    assert var_seq[snv_idx] == alt
    return ref_seq, var_seq, snv_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 0))

    outdir = Path(cfg["out_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    layers = cfg["layers"]  # e.g. ["blocks.24.mlp.l3", ...]
    radius = cfg.get("local_radius", 8)

    df = pd.read_parquet(cfg["table"])
    if cfg.get("categories"):
        df = df[df["consequence"].isin(cfg["categories"])].reset_index(drop=True)
    df = df.reset_index(drop=True)
    if cfg.get("limit"):
        df = df.iloc[: int(cfg["limit"])].reset_index(drop=True)
    n = len(df)
    print(f"Extracting Evo2 activations for {n} variants, layers={layers}")

    seq_chr17 = load_chr17(cfg["chr17_fasta"])

    from evo2 import Evo2
    model = Evo2(cfg.get("model", "evo2_7b"))

    D = cfg.get("hidden", 4096)
    store = {f"{l}_{k}": np.zeros((n, D), np.float32)
             for l in layers for k in ("wt_exact", "mut_exact", "wt_local", "mut_local")}
    ok = np.zeros(n, bool)

    ckpt = outdir / "evo2_store.npz"
    idx_path = outdir / "index.parquet"
    start_i = 0
    if ckpt.exists() and idx_path.exists() and not cfg.get("overwrite"):
        z = np.load(ckpt)
        for k in store:
            if k in z:
                store[k] = z[k]
        ok = z["ok"]
        start_i = int(ok.sum())
        print(f"Resuming from {start_i}/{n}")

    t0 = time.time()
    for i in range(start_i, n):
        r = df.iloc[i]
        try:
            ref_seq, var_seq, snv_idx = parse_sequences(seq_chr17, int(r.pos), r.ref, r.alt)
        except AssertionError as e:
            df.loc[i, "reason_code"] = str(e)
            continue
        for seq, tag in ((ref_seq, "wt"), (var_seq, "mut")):
            ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).cuda()
            with torch.inference_mode():
                _, emb = model(ids, return_embeddings=True, layer_names=layers)
            for l in layers:
                a = emb[l][0]  # (L, D)
                store[f"{l}_{tag}_exact"][i] = to_np(pool_exact(a, snv_idx))
                store[f"{l}_{tag}_local"][i] = to_np(pool_local_mean(a, snv_idx, radius))
        ok[i] = True
        if (i + 1) % 100 == 0 or i == n - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  ({dt/ max(1,(i+1-start_i)):.2f}s/var, {dt/60:.1f}min)", flush=True)
            np.savez(ckpt, ok=ok, **store)
            df.to_parquet(idx_path)

    np.savez(ckpt, ok=ok, **store)
    df.to_parquet(idx_path)
    save_json({"n": n, "n_ok": int(ok.sum()), "layers": layers, "model": cfg.get("model"),
               "window": WINDOW, "radius": radius}, outdir / "evo2_meta.json")
    print(f"Done. {int(ok.sum())}/{n} extracted -> {ckpt}")


if __name__ == "__main__":
    main()
