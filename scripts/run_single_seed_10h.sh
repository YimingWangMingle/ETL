#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
OUTPUT_ROOT="${SHORT_OUTPUT_ROOT:-runs/single_seed_10h}"
HAND_CONFIG="configs/single_seed_10h_hand.yaml"
LEG_CONFIG="configs/single_seed_10h_leg.yaml"
START_SECONDS=${SECONDS}

run_server() {
  python -m etl_sar.formal.server "$@" \
    --output-root "${OUTPUT_ROOT}" \
    --hand-config "${HAND_CONFIG}" \
    --leg-config "${LEG_CONFIG}"
}

aggregate_results() {
  python -m etl_sar.formal.aggregate_cli \
    --output-root "${OUTPUT_ROOT}" \
    --hand-config "${HAND_CONFIG}" \
    --leg-config "${LEG_CONFIG}"
}

case "${MODE}" in
  dry-run)
    run_server dry-run
    ;;
  run)
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required for the ten-hour profile"; print(f"CUDA device: {torch.cuda.get_device_name(0)}")'

    for index in 0 1; do
      run_server source --index "${index}"
    done

    for index in 0 1 2 3 4 5; do
      run_server target --index "${index}"
    done

    aggregate_results
    printf 'Single-seed comparison completed in %s seconds.\n' "$((SECONDS - START_SECONDS))"
    ;;
  aggregate)
    aggregate_results
    ;;
  *)
    printf 'usage: %s {dry-run|run|aggregate}\n' "$0"
    exit 2
    ;;
esac
