#!/usr/bin/env python3
"""Feasibility probe for Research C: does Evo2's forward support autograd for LoRA?

Loads Evo2, injects LoRA into a few Linear modules, enables grad on a short sequence,
and checks that a backward pass produces finite gradients on the LoRA parameters.
"""
import argparse

import torch

from cdd.finetune.lora import inject_lora, lora_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base")
    ap.add_argument("--target", default=r"blocks\.(1[0-9]|2[0-5])\.mlp\.l3")
    args = ap.parse_args()

    from evo2 import Evo2
    evo = Evo2(args.model)
    core = evo.model

    n_linear = sum(1 for _, m in core.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"Total nn.Linear modules: {n_linear}")
    wrapped = inject_lora(core, args.target, r=8, alpha=16)
    print(f"Wrapped {len(wrapped)} modules, e.g. {wrapped[:5]}")
    params = lora_parameters(core)
    print(f"LoRA trainable params: {sum(p.numel() for p in params):,}")

    seq = "ACGT" * 64
    ids = torch.tensor(evo.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).cuda()
    core.train()
    try:
        logits = core.forward(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = logits.float().pow(2).mean()
        loss.backward()
        gnorms = [(p.grad is not None and torch.isfinite(p.grad).all().item(),
                   float(p.grad.norm()) if p.grad is not None else None) for p in params]
        n_ok = sum(1 for ok, _ in gnorms if ok)
        print(f"BACKWARD OK. loss={loss.item():.4f}  params_with_finite_grad={n_ok}/{len(params)}")
        print("sample grad norms:", [round(g, 5) for _, g in gnorms[:5] if g is not None])
        print("C_FEASIBLE" if n_ok == len(params) and n_ok > 0 else "C_PARTIAL")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("C_INFEASIBLE:", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    main()
