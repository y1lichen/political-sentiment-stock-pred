#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-2330.TW}"
MARKET_MODEL="${2:-lightgbm}"
OVERLAY_MODEL="${3:-elasticnet}"
SPLIT="${4:-regime_aware}"

python -m src.training.train_overlay \
  --target "${TARGET}" \
  --market-model "${MARKET_MODEL}" \
  --overlay-model "${OVERLAY_MODEL}" \
  --split "${SPLIT}"

