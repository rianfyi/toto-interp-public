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
# Five-seed taxonomy-control GPU stage: activation dumps plus GPU raw-window controls.
# Submit after staging BOOM and the Toto checkpoint on the login node. Dispatch
# the two CPU stages with task-correlated dependencies so each seed advances
# independently:
#
#   gpu=$(sbatch --parsable examples/slurm/run_reviewer3_suite.sh)
#   cpu=$(sbatch --parsable --dependency=aftercorr:${gpu} examples/slurm/run_reviewer3_cpu_followup.sh)
#   sbatch --dependency=aftercorr:${cpu} examples/slurm/run_reviewer3_holdout_grid.sh
#
#SBATCH --job-name=toto-r3
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/reviewer3_%A_%a.out
#SBATCH --error=logs/reviewer3_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer3_evalmode_20260728}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
ROTATION="${STRUCTURAL_HOLDOUT_ROTATION:-$((SEED - 42))}"

module load ${SOFTWARE_STACK}
PYTHON="${VENV}/bin/python"
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
export HUGGING_FACE_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${REPO}/logs" "${RUNS_ROOT}"
echo "reviewer3 seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" "${REPO}/scripts/run_reviewer3_suite.py" \
  --output-root "${RUNS_ROOT}" \
  --seed "${SEED}" \
  --device cuda \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --context-length 1024 \
  --max-series-per-split 500 \
  --max-windows-per-series 4 \
  --raw-width 32 \
  --raw-layers 3 \
  --raw-modes 16 \
  --epochs 20 \
  --batch-size 16 \
  --structural-holdout-target frequency_bucket \
  --structural-holdout-mode combination \
  --structural-holdout-rotation "${ROTATION}" \
  --raw-only \
  ${REUSE_EXISTING:+--reuse-existing}

echo "reviewer3 seed=${SEED} end=$(date)"
