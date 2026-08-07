#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
OUTPUT_ROOT="${DMC_OUTPUT_ROOT:-runs/dmc_transfer_pilot}"

case "${MODE}" in
  dry-run|aggregate)
    python -m etl_sar.dmc.server "${MODE}" --output-root "${OUTPUT_ROOT}"
    ;;
  run)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required for this pilot"; print(f"CUDA device: {torch.cuda.get_device_name(0)}")'
    python -m etl_sar.dmc.server run --output-root "${OUTPUT_ROOT}" --device cuda
    ;;
  source|target)
    if [[ $# -ne 2 ]]; then
      printf 'usage: %s %s INDEX\n' "$0" "${MODE}"
      exit 2
    fi
    python -m etl_sar.dmc.server "${MODE}" --index "$2" --output-root "${OUTPUT_ROOT}" --device cuda
    ;;
  *)
    printf 'usage: %s {dry-run|run|aggregate|source INDEX|target INDEX}\n' "$0"
    exit 2
    ;;
esac
