#!/bin/bash
cd ~/neurips/central-dogma-diffing; source ~/evo/.venv/bin/activate; export PYTHONPATH=src
echo "=== LoRA fine-tune $(date) ==="
python3 scripts/train_evo2_lora.py --config configs/experiments/c_lora_mvp.yaml || { echo C_FAIL_LORA; exit 1; }
echo "=== matched extraction $(date) ==="
python3 scripts/extract_matched_activations.py --config configs/experiments/c_matched.yaml || { echo C_FAIL_MATCHED; exit 1; }
echo "=== delta crosscoder $(date) ==="
python3 scripts/train_c_delta_crosscoder.py --config configs/experiments/c_delta_mvp.yaml || { echo C_FAIL_DELTA; exit 1; }
echo "C_DONE $(date)"
