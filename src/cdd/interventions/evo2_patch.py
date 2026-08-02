"""Activation-patching interventions on Evo2 to causally test shared latents.

We run Evo2 on the variant window while a forward hook edits the chosen layer's
activation at the SNV token, then recompute the variant log-likelihood delta score
(LL(var) - LL(ref)). Removing a shared latent's decoder direction should move the
variant's score toward wild-type if that latent carries the functional signal.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def seq_loglik(model, ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Sum log P(token_t | <t) over the sequence (next-token, char-level)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    tgt = ids[0, 1:].long()
    lp = logp[0, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp.sum().item()


class Patcher:
    """Registers a hook on `layer_name` that adds `delta_vec` at `pos` on the fly."""

    def __init__(self, model, layer_name: str):
        self.model = model
        self.layer = model.model.get_submodule(layer_name)
        self.delta = None
        self.pos = None
        self.handle = None

    def _hook(self, _, __, output):
        if self.delta is None:
            return output
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h.clone()
        h[0, self.pos] = h[0, self.pos] + self.delta.to(h.dtype)
        return (h,) + output[1:] if is_tuple else h

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()

    @torch.no_grad()
    def scored_forward(self, ids, pos=None, delta_vec=None):
        self.pos = pos
        self.delta = delta_vec
        logits = self.model.model.forward(ids)
        self.delta = None
        return seq_loglik(self.model, ids, logits)


@torch.no_grad()
def variant_delta_score(model, tokenizer, ref_seq, var_seq, layer_name,
                        pos=None, patch_vec=None):
    """LL(var) - LL(ref); optionally patch the variant forward at `pos`."""
    ref_ids = torch.tensor(tokenizer.tokenize(ref_seq), dtype=torch.int).unsqueeze(0).cuda()
    var_ids = torch.tensor(tokenizer.tokenize(var_seq), dtype=torch.int).unsqueeze(0).cuda()
    with Patcher(model, layer_name) as p:
        ll_ref = p.scored_forward(ref_ids)  # no patch
        ll_var = p.scored_forward(var_ids, pos=pos, delta_vec=patch_vec)
    return ll_var - ll_ref
