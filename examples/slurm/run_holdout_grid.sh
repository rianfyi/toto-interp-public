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
# Taxonomy-control held-out cells not covered by the main CPU follow-up.
# One task per resplit loads each activation source once, then evaluates all
# four held-out target/mode cells plus conditional probes in-process.
#SBATCH --job-name=toto-r3-holdouts
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/holdout_grid_%A_%a.out
#SBATCH --error=logs/holdout_grid_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/taxonomy_suite}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
ROTATION="${STRUCTURAL_HOLDOUT_ROTATION:-$((SEED - 42))}"
SEED_ROOT="${RUNS_ROOT}/seed_${SEED}"

module load ${SOFTWARE_STACK}
for SOURCE in pretrained random_init; do
  "${VENV}/bin/python" "${REPO}/scripts/run_holdout_grid.py" \
    --activation-files \
      "${SEED_ROOT}/${SOURCE}_activations/train_activations.pt" \
      "${SEED_ROOT}/${SOURCE}_activations/val_activations.pt" \
      "${SEED_ROOT}/${SOURCE}_activations/test_activations.pt" \
    --output-root "${SEED_ROOT}/structural_holdout" \
    --conditional-output-dir "${SEED_ROOT}/conditional/${SOURCE}" \
    --source "${SOURCE}" \
    --seed "${SEED}" \
    --rotation "${ROTATION}" \
    --reuse-existing
done
