# CentralDogma-ΔCC / DeltaEvo

Sparse **variant-delta crosscoders** that connect the internal representations of a
genomic language model (**Evo 2**) and a protein language model (**ESM-2**), and that
diff a base vs. disease-fine-tuned Evo 2.

Two studies (see `RESEARCH_PLAN.md`):

- **B — CentralDogma-ΔCC:** For a missense variant we compute the *paired* activation
  delta in each model, `Δh_DNA = h_DNA(mut) − h_DNA(wt)` and `Δh_PROT = h_PROT(mut) − h_PROT(wt)`,
  and learn a shared + modality-private sparse crosscoder over the pair. We evaluate on
  BRCA1 saturation genome editing (Findlay et al. 2018): cross-modal retrieval, DMS-score
  and ClinVar prediction, latent interpretability, and causal ablation/injection.
- **C — DeltaEvo:** Diff base vs. LoRA-fine-tuned Evo 2 with a Delta-Crosscoder.

## Environment

Uses the existing Evo 2 venv (`~/evo/.venv`): torch cu128, `evo2`, `transformers` (ESM-2),
`biopython`, `scikit-learn`. Single A100 80GB.

```bash
source ~/evo/.venv/bin/activate
export PYTHONPATH=src
```

## Pipeline

```bash
python scripts/build_variant_table.py    --config configs/data/brca1_mvp.yaml
python scripts/extract_evo2_activations.py --config configs/experiments/brca1_evo2.yaml
python scripts/extract_esm_activations.py  --config configs/experiments/brca1_esm.yaml
python scripts/train_b_crosscoder.py       --config configs/experiments/b_mvp.yaml
python scripts/evaluate_b.py               --run-dir outputs/b_mvp
```

Status and results are tracked in `RESULTS.md`.
