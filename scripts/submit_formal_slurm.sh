#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/slurm

SOURCE_JOB_ID=$(sbatch --parsable \
  --job-name=etl-source \
  --array=0-9%5 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=48:00:00 \
  --output=logs/slurm/source-%A_%a.out \
  --wrap="bash scripts/run_formal_server.sh source")

sbatch \
  --job-name=etl-lattice-target \
  --array=0-29%6 \
  --dependency="afterok:${SOURCE_JOB_ID}" \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=120:00:00 \
  --output=logs/slurm/target-%A_%a.out \
  --wrap="bash scripts/run_formal_server.sh target"
