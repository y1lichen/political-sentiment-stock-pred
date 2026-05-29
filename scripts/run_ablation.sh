#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-2330.TW}"
MODEL="${2:-event_gated_mlp}"
SPLIT="${3:-regime_aware}"

FEATURE_SETS=(
  TW_self_only
  TW_market_only
  TW_plus_global_market
  Trump_text_only
  TW_plus_Trump
  Global_plus_Trump_no_gate
  Global_plus_Trump_with_gate
)

for FEATURE_SET in "${FEATURE_SETS[@]}"; do
  python -m src.training.train \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --feature-set "${FEATURE_SET}" \
    --all-features
done

python -m src.evaluation.run_ablation_diagnostics \
  --target "${TARGET}" \
  --model "${MODEL}" \
  --split "${SPLIT}"
