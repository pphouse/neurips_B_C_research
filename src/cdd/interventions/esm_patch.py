"""Activation-patching interventions on ESM-2 (masked-marginal variant scoring).

Variant effect score (masked-marginal): mask the variant residue, read the LM-head
log-probs, score = logP(mut_aa) - logP(wt_aa). A forward hook can add a direction to
the hidden state at the variant residue at a chosen encoder layer, so we can ablate a
shared latent's protein decoder direction and measure the score shift.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class EsmScorer:
    def __init__(self, model_name: str, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=torch.float32).to(device).eval()
        self.device = device
        # locate encoder layers module list
        self.layers = self.model.esm.encoder.layer

    def _enc(self, seq):
        e = self.tok(seq, return_tensors="pt", add_special_tokens=True)
        return {k: v.to(self.device) for k, v in e.items()}

    @torch.no_grad()
    def masked_marginal(self, seq, res_idx, wt_aa, mut_aa, layer_idx=None, delta_vec=None):
        """res_idx: 0-based residue index in `seq`. Token position = res_idx+1 (CLS).
        Optionally patch hidden state at that token after encoder layer `layer_idx-1`."""
        enc = self._enc(seq)
        tok_pos = res_idx + 1
        enc["input_ids"][0, tok_pos] = self.tok.mask_token_id
        handle = None
        if delta_vec is not None and layer_idx is not None:
            mod = self.layers[layer_idx - 1]

            def hook(_, __, output):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                h[0, tok_pos] = h[0, tok_pos] + delta_vec.to(h.dtype)
                return (h,) + output[1:] if isinstance(output, tuple) else h

            handle = mod.register_forward_hook(hook)
        try:
            out = self.model(**enc)
        finally:
            if handle:
                handle.remove()
        logp = F.log_softmax(out.logits[0, tok_pos].float(), -1)
        wt_id = self.tok.convert_tokens_to_ids(wt_aa)
        mut_id = self.tok.convert_tokens_to_ids(mut_aa)
        return (logp[mut_id] - logp[wt_id]).item()
