#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-2330.TW}"
MODEL_PATH="${2:?Usage: scripts/predict.sh TARGET MODEL_PATH [latest|all]}"
MODE="${3:-latest}"

LATEST_FLAG=""
if [[ "${MODE}" == "latest" ]]; then
  LATEST_FLAG="--latest"
fi

python -m src.inference.predict \
  --target "${TARGET}" \
  --model-path "${MODEL_PATH}" \
  ${LATEST_FLAG}

