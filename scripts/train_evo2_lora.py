#!/usr/bin/env python3
"""Research C: LoRA fine-tune Evo2 on BRCA1 LOF-vs-FUNC classification.

Backbone frozen; we train LoRA adapters on late-middle layers plus a small head that reads
the variant-position delta activation (pool(mut) - pool(wt)) at a chosen layer. Supports a
label-permutation negative control and a gene/domain-disjoint OOD split.
"""
import argparse
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.metrics import roc_auc_score

from cdd.finetune.lora import inject_lora, lora_parameters, defuse_inference_tensors
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
    assert ref_seq[idx] == ref
    return ref_seq, var_seq, idx


class ActHook:
    """Capture a layer's output WITH grad during forward."""
    def __init__(self, model, layer_name):
        self.h = None
        self.layer = model.get_submodule(layer_name)
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, _, __, out):
        self.h = out[0] if isinstance(out, tuple) else out
    def close(self):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 0))
    dev = "cuda"
    run_dir = Path(cfg["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg["table"])
    df = df[df.is_missense & df.func_class.isin(["LOF", "FUNC"])].reset_index(drop=True)
    if cfg.get("limit"):
        df = df.iloc[: int(cfg["limit"])].reset_index(drop=True)
    y = (df.func_class == "LOF").astype(np.float32).to_numpy()
    if cfg.get("permute_labels"):
        rng = np.random.default_rng(cfg.get("seed", 0)); y = rng.permutation(y)
    split = df[cfg.get("split_col", "split_position")].to_numpy()
    tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]
    print(f"C LoRA: n={len(df)} train={len(tr)} test={len(te)} pos_rate={y.mean():.2f} "
          f"permute={bool(cfg.get('permute_labels'))}")

    seq_chr17 = load_chr17(cfg["chr17_fasta"])
    W = cfg.get("window_bp", 1536)
    layer = cfg["layer"]

    from evo2 import Evo2
    evo = Evo2(cfg.get("model", "evo2_7b")); core = evo.model
    defuse_inference_tensors(core)
    wrapped = inject_lora(core, cfg["lora_target"], r=cfg.get("rank", 8),
                          alpha=cfg.get("alpha", 16), dropout=cfg.get("dropout", 0.0))
    core.cuda(); core.train()
    print(f"LoRA wrapped {len(wrapped)} modules")

    head = torch.nn.Sequential(
        torch.nn.LayerNorm(cfg.get("hidden", 4096)),
        torch.nn.Linear(cfg.get("hidden", 4096), 1)).cuda()
    params = lora_parameters(core) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 0.01))
    hook = ActHook(core, layer)

    def variant_logit(pos, ref, alt):
        ref_seq, var_seq, idx = make_window(seq_chr17, pos, ref, alt, W)
        outs = []
        for s in (ref_seq, var_seq):
            ids = torch.tensor(evo.tokenizer.tokenize(s), dtype=torch.int).unsqueeze(0).cuda()
            core.forward(ids)
            outs.append(hook.h[0, idx].float())  # (D,) with grad
        delta = outs[1] - outs[0]
        return head(delta.unsqueeze(0)).squeeze()

    epochs = cfg.get("epochs", 3)
    t0 = time.time(); log = []
    for ep in range(epochs):
        perm = np.random.permutation(tr)
        core.train()
        for step, i in enumerate(perm):
            r = df.iloc[i]
            logit = variant_logit(int(r.pos), r.ref, r.alt)
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[i], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
        core.eval()
        with torch.no_grad():
            def scores(idxs):
                out = []
                for i in idxs:
                    r = df.iloc[i]
                    out.append(torch.sigmoid(variant_logit(int(r.pos), r.ref, r.alt)).item())
                return np.array(out)
            str_ = scores(tr); ste = scores(te)
        au_tr = roc_auc_score(y[tr], str_) if len(np.unique(y[tr])) > 1 else float("nan")
        au_te = roc_auc_score(y[te], ste) if len(np.unique(y[te])) > 1 else float("nan")
        rec = dict(epoch=ep, auroc_train=au_tr, auroc_test=au_te, min=(time.time()-t0)/60)
        log.append(rec)
        print(f"  epoch {ep}: AUROC train={au_tr:.3f} test(OOD)={au_te:.3f} ({rec['min']:.1f}min)", flush=True)

    hook.close()
    torch.save({"lora": {n: p.detach().cpu() for n, p in core.named_parameters() if ".A" in n or ".B" in n},
                "head": head.state_dict(), "wrapped": wrapped, "cfg": cfg},
               run_dir / "lora_ckpt.pt")
    save_json(log, run_dir / "lora_log.json")
    save_json({"wrapped": wrapped, "n_train": len(tr), "n_test": len(te),
               "final": log[-1] if log else None}, run_dir / "lora_meta.json")
    print(f"Saved LoRA -> {run_dir}")


if __name__ == "__main__":
    main()
