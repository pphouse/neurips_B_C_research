# Results log

Final numbers are generated into the paper via `scripts/make_paper_results.py` from the JSONs
under `outputs/`. All B numbers are 5-seed unless noted.

## Data
- BRCA1 SGE (Findlay 2018): 3893 SNVs; **2086 missense** paired (Evo2 DNA-delta + ESM-2 protein-delta).
- Protein mapping validated vs UniProt P38398 (0/2086 mismatches).
- Domains: RING (642 missense), BRCT (1363). Splits: residue-disjoint (n_test=366), domain-disjoint BRCT→RING (n_test=642).

## Configuration (selected by DMS layer probe)
- DNA: Evo2-7B `blocks.24.mlp.l3`, local-mean pooling (±8). Protein: ESM-2 `esm2_t33_650M` layer 33.
- Crosscoder: PCA-denoise deltas to 128 dims; k_shared=32, k_private=96, d_align=64, BatchTopK L0=24.
- Key design: a **linear alignment head** (deep-CCA-style) carries the shared representation for
  retrieval/probing; sparse codes are the reconstruction dictionary. (Pure ReLU-shared alignment
  mode-collapses.) Seed set before model init (retrieval quality is init-sensitive).

## Research B — headline results (5 seeds)
| metric | crosscoder (ours) | CCA | DNA Δ | ESM Δ | concat |
|---|---|---|---|---|---|
| Retrieval R@1 (residue) | **0.282±0.005** | 0.219 | – | – | – |
| Retrieval R@10 (residue) | **0.689** | 0.642 | – | – | – |
| Retrieval R@1 (domain BRCT→RING) | **0.117** | 0.076 | – | – | – |
| DMS Spearman (residue) | 0.459±0.006 | 0.335 | 0.375 | 0.510 | 0.512 |
| DMS Spearman (domain) | 0.302 | 0.281 | 0.252 | 0.447 | 0.389 |
| Reconstruction FVE | DNA 0.72 / PROT 0.70 | – | – | – | – |

**Reading:** the shared representation beats the linear cross-modal baseline (CCA) and the DNA
probe on both retrieval and DMS, and generalizes to a held-out domain; ESM-2 alone / concat are
stronger DMS predictors (ESM captures missense well) — the shared code's value is interpretability
and cross-modal alignment, not raw accuracy.

## Biological structure (collective probing, held-out)
- Shared rep predicts LOF-vs-FUNC AUROC **0.838**, RING-vs-BRCT 0.787.
- Protein-private best predicts domain (0.848) and LOF (0.822); DNA-private predicts LOF (0.744).

## Shared functional axis (control)
- cos(DNA-functional dir, PROT-functional dir) in shared space = **0.852**
- cos(domain, domain) = 0.838; cos(functional, domain) = **−0.267**; random pair 0.075±0.17 (p95 |cos| 0.38).
- → the shared space organizes biology consistently across modalities, specifically (not an alignment artifact).

## Causal probing (honest, mixed)
- Injecting the shared functional direction shifts ESM-2 masked-marginal score DMS-correlated (Spearman 0.257).
- The same injection into Evo2's likelihood is non-specific (≈ random-direction control).
- → representational alignment does NOT imply an interchangeable causal handle (reported as a caution).

## Research C — DeltaEvo (in progress / see outputs/c_*)
- Evo2 LoRA autograd feasible via `defuse_inference_tensors` (clone inference-tensor buffers).
- LoRA fine-tune (LOF-vs-FUNC, BRCT→RING OOD) + matched base/ft activation extraction + Delta-Crosscoder taxonomy.
