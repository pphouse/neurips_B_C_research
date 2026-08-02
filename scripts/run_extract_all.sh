#!/bin/bash
set -e
cd ~/neurips/central-dogma-diffing
source ~/evo/.venv/bin/activate
export PYTHONPATH=src
echo "=== ESM full $(date) ==="
python3 scripts/extract_esm_activations.py --config configs/experiments/brca1_esm.yaml
echo "=== EVO2 full $(date) ==="
python3 scripts/extract_evo2_activations.py --config configs/experiments/brca1_evo2.yaml
echo "=== ALL DONE $(date) ==="
