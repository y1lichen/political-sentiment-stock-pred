#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-2330.TW}"
python -m src.data.build_dataset --target "${TARGET}"

