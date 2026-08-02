#!/usr/bin/env python3
"""Research C: extract matched base vs. LoRA-fine-tuned Evo2 variant deltas.

For each missense variant we compute Δh = pool(mut) - pool(wt) at a chosen layer, once with
the base model and once with the fine-tuned (LoRA) model, on identical windows. These paired
deltas are the input to the Delta-Crosscoder.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from cdd.finetune.lora import inject_lora, defuse_inference_tensors
from cdd.utils.common import load_yaml, save_json, set_seed


def load_chr17(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return str(list(SeqIO.parse(f, "fasta"))[0].seq)


def make_window(seq_chr17, pos, ref, alt, W):
    p = pos - 1
    s = max(0, p - W // 2); e = min(len(seq_chr17), p + W // 2)
    ref_seq = seq_chr17[s:e]; idx = min(W // 2, p)
    var_seq = ref_seq[:idx] + alt + ref_seq[idx + 1:]
    return ref_seq, var_seq, idx


@torch.no_grad()
def extract(core, evo, df, seq_chr17, layer, W, dev, bs=8):
    from cdd.crosscoder.model import batch_topk  # noqa (ensure import path ok)
    hook_store = {}
    mod = core.get_submodule(layer)
    def hook(_, __, out):
        hook_store["h"] = (out[0] if isinstance(out, tuple) else out)
    handle = mod.register_forward_hook(hook)
    D = None
    deltas = []
    for k in range(0, len(df), bs):
        rows = df.iloc[k:k + bs]
        wt_ids, mut_ids, idxs = [], [], []
        for _, r in rows.iterrows():
            ref_seq, var_seq, idx = make_window(seq_chr17, int(r.pos), r.ref, r.alt, W)
            wt_ids.append(evo.tokenizer.tokenize(ref_seq))
            mut_ids.append(evo.tokenizer.tokenize(var_seq)); idxs.append(idx)
        ids = torch.tensor(wt_ids + mut_ids, dtype=torch.int).cuda()
        core.forward(ids)
        h = hook_store["h"].float()
        B = len(rows); pos = torch.tensor(idxs, device=dev)
        wt = h[torch.arange(B), pos]; mut = h[torch.arange(B, 2 * B), pos]
        deltas.append((mut - wt).cpu().numpy())
    handle.remove()
    return np.concatenate(deltas, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 0))
    dev = "cuda"
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg["table"])
    df = df[df.is_missense & df.func_class.isin(["LOF", "FUNC"])].reset_index(drop=True)
    if cfg.get("limit"):
        df = df.iloc[: int(cfg["limit"])].reset_index(drop=True)
    seq_chr17 = load_chr17(cfg["chr17_fasta"])
    W = cfg.get("window_bp", 1024); layer = cfg["layer"]

    from evo2 import Evo2
    # base model (inference is fine)
    evo = Evo2(cfg.get("model", "evo2_7b")); core = evo.model
    print("Extracting BASE deltas...")
    base = extract(core, evo, df, seq_chr17, layer, W, dev, cfg.get("batch_variants", 8))

    # fine-tuned model: defuse, inject LoRA, load trained weights
    print("Loading LoRA and extracting FT deltas...")
    ck = torch.load(cfg["lora_ckpt"], map_location="cpu", weights_only=False)
    defuse_inference_tensors(core)
    inject_lora(core, ck["cfg"]["lora_target"], r=ck["cfg"].get("rank", 8),
                alpha=ck["cfg"].get("alpha", 16))
    core.cuda()
    sd = {n: p for n, p in core.named_parameters()}
    for n, w in ck["lora"].items():
        if n in sd:
            sd[n].data.copy_(w.to(sd[n].dtype).to(dev))
    core.eval()
    ft = extract(core, evo, df, seq_chr17, layer, W, dev, cfg.get("batch_variants", 8))

    np.savez(out / "matched.npz", base=base, ft=ft,
             y=(df.func_class == "LOF").astype(np.int8).to_numpy())
    df.to_parquet(out / "index.parquet")
    save_json({"n": len(df), "layer": layer, "window": W, "dim": int(base.shape[1]),
               "mean_delta_model_diff_L2": float(np.linalg.norm(ft - base, axis=1).mean())},
              out / "matched_meta.json")
    print(f"Saved matched deltas {base.shape} -> {out}. "
          f"mean ||Δft - Δbase|| = {np.linalg.norm(ft - base, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
