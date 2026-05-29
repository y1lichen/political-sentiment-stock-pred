#!/usr/bin/env bash
set -euo pipefail

MARKET_MODEL="${1:-lightgbm}"
OVERLAY_MODEL="${2:-elasticnet}"
SPLIT="${3:-regime_aware}"
TRANSACTION_COST="${4:-0.001}"
SLIPPAGE="${5:-0.0005}"

python -m src.training.train_overlay_all \
  --market-model "${MARKET_MODEL}" \
  --overlay-model "${OVERLAY_MODEL}" \
  --split "${SPLIT}" \
  --transaction-cost "${TRANSACTION_COST}" \
  --slippage "${SLIPPAGE}"
