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
# Fit all dynamic TOTO probes on five corrected eval-mode BOOM resplits.
#SBATCH --job-name=toto-dyn5
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/reviewer_rep_toto_dynamic_%A_%a.out
#SBATCH --error=logs/reviewer_rep_toto_dynamic_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO}/runs/reviewer3_evalmode_20260728}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_evalmode_20260728}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

ACT_ROOT="${SOURCE_ROOT}/seed_${SEED}/pretrained_activations"
OUTPUT_ROOT="${RUNS_ROOT}/seed_${SEED}/toto_dynamic_probes"

echo "toto dynamic probes seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" "${REPO}/scripts/fit_toto_probes.py" \
  --activation-files \
    "${ACT_ROOT}/train_activations.pt" \
    "${ACT_ROOT}/val_activations.pt" \
    "${ACT_ROOT}/test_activations.pt" \
  --output-dir "${OUTPUT_ROOT}" \
  --label-group dynamic \
  --method linear_probe \
  --seed "${SEED}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --reuse-artifacts

echo "toto dynamic probes seed=${SEED} end=$(date)"
