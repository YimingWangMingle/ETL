#!/usr/bin/env bash
set -euo pipefail

python -m etl_sar.formal.aggregate_cli \
  --output-root "${FORMAL_OUTPUT_ROOT:-runs/formal}"
