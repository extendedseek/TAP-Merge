#!/usr/bin/env bash
set -euo pipefail
python -m tap_merge smoke --output "${1:-runs/smoke}" --all-ablations
