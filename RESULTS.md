# Results log

Living record of experiments. Final numbers are regenerated on the full dataset and
injected into the paper via `scripts/make_paper_results.py`.

## Data
- BRCA1 SGE (Findlay 2018): 3893 SNVs; **2086 missense** paired (Evo2 DNA-delta + ESM-2 protein-delta).
- Protein mapping validated against UniProt P38398 (0/2086 mismatches).
- Domains: RING (642 missense), BRCT (1363), linker (81). Splits: residue-disjoint, domain-disjoint (train BRCT → test RING).

## Layer selection (DMS probe)
- Best DNA layer: **blocks.26.mlp.l3** (delta-activation ridge Spearman ≈0.37 on residue-disjoint test).
- Best protein layer: **ESM-2 L18** (Spearman ≈0.65); L30/L33 comparable.
- blocks.28+ have exploding activation norms (StripedHyena massive activations) — avoided.

## Method notes (validated on data)
- Variant deltas are high intrinsic dimension (~50 PCs for 90% var) → PCA-denoise (n_pca=96–128) before the crosscoder, else held-out reconstruction FVE is ~0.
- Plain shared-alignment MSE causes mode collapse; contrastive InfoNCE (with tiny align-MSE) is required for discriminative aligned shared codes.

## Preliminary results (PARTIAL data, n≈297 paired, residue-disjoint; will be superseded)
- Reconstruction: held-out FVE_dna ≈0.2–0.5, FVE_prot ≈0.2–0.5 after PCA denoising.
- DMS Spearman: shared-code 0.52 > CCA 0.40, PLS 0.43, DNA-only 0.46; < concat 0.70, protein-only 0.71.
  → shared code beats *linear cross-modal* baselines; concat/protein are upper bounds.
- Cross-modal retrieval: crosscoder currently ties/below CCA at small n (overfitting); expect improvement at full n=2086.
- External predictors (|Spearman| vs DMS): SIFT 0.37, CADD 0.24, phyloP 0.32 — LM deltas (0.46/0.71) exceed them.

## TODO on full data
- Re-run layer probe, train crosscoder, tune contrast weight.
- Retrieval vs CCA at full n; DMS residue- and domain-disjoint.
- ClinVar AUROC (more labels available at full n).
- Interpretability: enriched latent counts; per-latent domain/LOF profiles.
- Cross-model causal ablation (headline).
- Research C feasibility (Evo2 LoRA autograd).
