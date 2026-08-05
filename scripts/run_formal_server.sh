#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
INDEX="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-runs/formal}"

case "${MODE}" in
  dry-run)
    python -m etl_sar.formal.server dry-run \
      --output-root "${OUTPUT_ROOT}"
    ;;
  source|target)
    python -m etl_sar.formal.server "${MODE}" \
      --index "${INDEX}" \
      --output-root "${OUTPUT_ROOT}"
    ;;
  *)
    echo "usage: $0 {dry-run|source|target} [array-index]" >&2
    exit 2
    ;;
esac
