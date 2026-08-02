#!/bin/bash
cd ~/neurips/central-dogma-diffing
source ~/evo/.venv/bin/activate
export PYTHONPATH=src
python3 scripts/extract_evo2_activations.py --config configs/experiments/brca1_evo2.yaml
echo "EVO2_DONE rc=$?"
