#!/bin/bash
# Site settings. Submit from the repository root after `mkdir -p logs`, passing
# your Slurm account and partition on every sbatch call, including the chained
# examples below (sbatch does not expand variables inside #SBATCH lines):
#   sbatch --account=<account> --partition=<partition> examples/slurm/<this file>
# GPU launchers request --gres=gpu:A10:1; override it on the command line
# (e.g. --gres=gpu:1) if your site names GPU types differently.
# Export before submitting:
#   REPO            absolute path of this repository checkout (falls back to SCRATCH)
#   VENV            Python 3.12 environment (default: ${REPO}/.venv-gpu)
#   SOFTWARE_STACK  environment module that provides Python
# Other ${VAR:-default} values below are functional defaults.
# CPU-only follow-up for probe fitting after GPU activation dumps are complete.
#SBATCH --job-name=toto-r3-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/taxonomy_probes_%A_%a.out
#SBATCH --error=logs/taxonomy_probes_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/taxonomy_suite}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
ROTATION="${STRUCTURAL_HOLDOUT_ROTATION:-$((SEED - 42))}"
HOLDOUT_TARGET="${STRUCTURAL_HOLDOUT_TARGET:-frequency_bucket}"
HOLDOUT_MODE="${STRUCTURAL_HOLDOUT_MODE:-combination}"

module load ${SOFTWARE_STACK}
"${VENV}/bin/python" "${REPO}/scripts/run_taxonomy_probes.py" \
  --runs-root "${RUNS_ROOT}" \
  --seed "${SEED}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --raw-width 32 \
  --raw-layers 3 \
  --raw-modes 16 \
  --epochs 20 \
  --batch-size 16 \
  --structural-holdout-target "${HOLDOUT_TARGET}" \
  --structural-holdout-mode "${HOLDOUT_MODE}" \
  --structural-holdout-rotation "${ROTATION}" \
  --skip-post-controls \
  --reuse-existing
