#!/usr/bin/env python3
"""Extract Evo2 (genomic-window) or ESM-2 (protein-window) variant deltas for the multi-gene
ClinVar table, using GRCh38 random access via pyfaidx. Mirrors the BRCA1 extractors
(blocks.24.mlp.l3 local pooling for Evo2; ESM-2 layer 33 local pooling) so the deltas live in
the same spaces. Resumable, batched for Evo2."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdd.activations.pooling import pool_local_mean, to_np
from cdd.utils.common import load_yaml, save_json, set_seed

WIN_DNA = 8192


def evo2_extract(cfg, df, dev):
    from pyfaidx import Fasta
    from evo2 import Evo2
    genome = Fasta(cfg["genome"], sequence_always_upper=True, rebuild=False)
    layer = cfg["dna_layer"]; radius = cfg.get("local_radius", 8)
    outdir = Path(cfg["out_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    n = len(df); D = 4096
    store = {f"wt": np.zeros((n, D), np.float32), "mut": np.zeros((n, D), np.float32)}
    ok = np.zeros(n, bool)
    ckpt = outdir / "evo2_store.npz"
    start = 0
    if ckpt.exists() and not cfg.get("overwrite"):
        z = np.load(ckpt); store["wt"], store["mut"], ok = z["wt"], z["mut"], z["ok"]
        start = int(ok.sum()); print(f"resume from {start}")
    model = Evo2(cfg.get("model", "evo2_7b"))
    vb = cfg.get("var_batch", 3)
    t0 = time.time(); i = start
    while i < n:
        rows = []; seqs = []; idxs = []
        j = i
        while j < n and len(rows) < vb:
            r = df.iloc[j]; p0 = int(r.pos) - 1
            s = max(0, p0 - WIN_DNA // 2); e = p0 + WIN_DNA // 2
            try:
                seq = str(genome[str(r.chrom)][s:e])
            except Exception:
                j += 1; continue
            idx = p0 - s
            if idx >= len(seq) or seq[idx] != r.ref:
                j += 1; continue  # ref mismatch or out of range
            var = seq[:idx] + r.alt + seq[idx + 1:]
            rows.append(j); idxs.append(idx)
            seqs.append(model.tokenizer.tokenize(seq)); seqs.append(model.tokenizer.tokenize(var))
            j += 1
        if rows:
            L = min(len(s) for s in seqs)
            seqs = [s[:L] for s in seqs]  # equal length within batch (windows equal unless near chrom end)
            ids = torch.tensor(seqs, dtype=torch.int).cuda()
            with torch.inference_mode():
                _, emb = model(ids, return_embeddings=True, layer_names=[layer])
            a = emb[layer]
            for bp, (row, idx) in enumerate(zip(rows, idxs)):
                idx = min(idx, a.shape[1] - 1)
                store["wt"][row] = to_np(pool_local_mean(a[2 * bp], idx, radius))
                store["mut"][row] = to_np(pool_local_mean(a[2 * bp + 1], idx, radius))
                ok[row] = True
        i = j
        if int(ok.sum()) % 150 < vb or i >= n:
            print(f"  {i}/{n} ok={int(ok.sum())} ({(time.time()-t0)/60:.1f}min)", flush=True)
            np.savez(ckpt, ok=ok, **store); df.to_parquet(outdir / "index.parquet")
    np.savez(ckpt, ok=ok, **store); df.to_parquet(outdir / "index.parquet")
    save_json({"n": n, "n_ok": int(ok.sum()), "layer": layer}, outdir / "meta.json")
    print(f"Evo2 done {int(ok.sum())}/{n}")


def esm_extract(cfg, df, dev):
    from transformers import AutoTokenizer, AutoModel
    from Bio import SeqIO
    layer = int(cfg["prot_layer"][1:]); radius = cfg.get("local_radius", 8)
    outdir = Path(cfg["out_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    proteins = {}
    for fa in Path(cfg["protein_dir"]).glob("*.fasta"):
        proteins[fa.stem] = str(list(SeqIO.parse(str(fa), "fasta"))[0].seq)
    tok = AutoTokenizer.from_pretrained(cfg["esm_model"])
    model = AutoModel.from_pretrained(cfg["esm_model"], torch_dtype=torch.float32).cuda().eval()
    D = model.config.hidden_size; n = len(df); ML = 1021
    store = {"wt": np.zeros((n, D), np.float32), "mut": np.zeros((n, D), np.float32)}
    ok = np.zeros(n, bool)
    t0 = time.time()
    for i in range(n):
        r = df.iloc[i]; prot = proteins.get(r.gene)
        if prot is None:
            continue
        p = int(r.aa_pos) - 1; s = max(0, p - 510); e = min(len(prot), s + ML); s = max(0, e - ML)
        wt = prot[s:e]; idx = p - s
        if idx >= len(wt) or wt[idx] != r.aa_ref:
            continue
        mut = wt[:idx] + r.aa_alt + wt[idx + 1:]
        for tag, seq in (("wt", wt), ("mut", mut)):
            enc = tok(seq, return_tensors="pt", add_special_tokens=True)
            with torch.inference_mode():
                out = model(**{k: v.cuda() for k, v in enc.items()}, output_hidden_states=True)
            store[tag][i] = to_np(pool_local_mean(out.hidden_states[layer][0], idx + 1, radius))
        ok[i] = True
        if (i + 1) % 300 == 0 or i == n - 1:
            print(f"  {i+1}/{n} ({(time.time()-t0)/60:.1f}min)", flush=True)
            np.savez(outdir / "esm_store.npz", ok=ok, **store); df.to_parquet(outdir / "index.parquet")
    np.savez(outdir / "esm_store.npz", ok=ok, **store); df.to_parquet(outdir / "index.parquet")
    save_json({"n": n, "n_ok": int(ok.sum()), "layer": layer}, outdir / "meta.json")
    print(f"ESM done {int(ok.sum())}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["evo2", "esm"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config); set_seed(0)
    df = pd.read_parquet(cfg["table"])
    if cfg.get("limit"):
        df = df.iloc[: int(cfg["limit"])].reset_index(drop=True)
    (evo2_extract if args.model == "evo2" else esm_extract)(cfg, df, "cuda")


if __name__ == "__main__":
    main()
