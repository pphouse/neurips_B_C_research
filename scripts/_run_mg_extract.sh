#!/bin/bash
cd ~/neurips/central-dogma-diffing; source ~/evo/.venv/bin/activate; export PYTHONPATH=src
echo "=== MG ESM $(date) ==="
python3 scripts/extract_multigene.py --model esm --config configs/experiments/mg_esm.yaml || { echo MG_FAIL_ESM; exit 1; }
echo "=== MG EVO2 $(date) ==="
python3 scripts/extract_multigene.py --model evo2 --config configs/experiments/mg_evo2.yaml || { echo MG_FAIL_EVO2; exit 1; }
echo "MG_DONE $(date)"
