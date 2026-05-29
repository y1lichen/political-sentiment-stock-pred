#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-2330.TW}"
MODEL="${2:-event_gated_mlp}"
SPLIT="${3:-regime_aware}"

python -m src.training.train \
  --target "${TARGET}" \
  --model "${MODEL}" \
  --split "${SPLIT}" \
  --rebuild-dataset

