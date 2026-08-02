#!/bin/bash
# Full Research B analysis on the complete activation set. Run after extraction finishes.
set -e
cd ~/neurips/central-dogma-diffing
source ~/evo/.venv/bin/activate
export PYTHONPATH=src

echo "=== [1/6] Layer probe (full data) ==="
python3 scripts/layer_probe.py --config configs/experiments/layer_probe.yaml

echo "=== [2/6] Train position-split crosscoder ==="
python3 scripts/train_b_crosscoder.py --config configs/experiments/b_mvp.yaml

echo "=== [3/6] Evaluate position split ==="
python3 scripts/evaluate_b.py --run-dir outputs/b_mvp --config configs/experiments/b_mvp.yaml

echo "=== [4/6] Train + eval domain-disjoint crosscoder (BRCT->RING) ==="
python3 scripts/train_b_crosscoder.py --config configs/experiments/b_domain.yaml
python3 scripts/evaluate_b.py --run-dir outputs/b_domain --config configs/experiments/b_domain.yaml

echo "=== [5/6] Interpretability latent analysis + figures ==="
python3 scripts/interpret_latents.py --run-dir outputs/b_mvp --config configs/experiments/b_mvp.yaml
python3 scripts/make_figures.py --run-dir outputs/b_mvp

echo "=== [6/6] Done. Key numbers: ==="
python3 - <<'PY'
import json
e=json.load(open("outputs/b_mvp/eval_b.json"))["split_position"]
print("retrieval CC R@1/R@10:", round(e["retrieval_crosscoder"]["R@1"],3), round(e["retrieval_crosscoder"]["R@10"],3),
      "| CCA:", round(e["retrieval_cca"]["R@1"],3), round(e["retrieval_cca"]["R@10"],3))
print("DMS:", {k:(round(v,3) if v==v else None) for k,v in e["dms"].items()})
PY
echo "B_FULL_DONE"
