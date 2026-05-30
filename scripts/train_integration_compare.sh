#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

TARGETS="${1:-default}"
MODEL="${2:-event_gated_mlp}"
SPLIT="${3:-regime_aware}"

"${PYTHON_BIN}" -m src.training.train_integration_compare \
  --targets "${TARGETS}" \
  --model "${MODEL}" \
  --split "${SPLIT}"
