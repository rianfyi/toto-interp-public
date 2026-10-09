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
# MOMENT-base random-init activation dumps for five BOOM resplits.
# Follow with the seed-correlated CPU probe/provenance stage:
#
#   gpu=$(sbatch --parsable examples/slurm/run_moment_random.sh)
#   sbatch --dependency=aftercorr:${gpu} examples/slurm/run_moment_random_probe_followup.sh
#SBATCH --job-name=moment-rand5
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/moment_random_%A_%a.out
#SBATCH --error=logs/moment_random_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/moment_suite}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
export HUGGING_FACE_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEED_ROOT="${RUNS_ROOT}/seed_${SEED}/moment_random"
ACT_ROOT="${SEED_ROOT}/activations"
mkdir -p "${REPO}/logs" "${SEED_ROOT}"

echo "moment random replication seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" "${REPO}/scripts/dump_moment_activations.py" \
  --output-dir "${ACT_ROOT}" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --device cuda \
  --weight-source random_init \
  --seed "${SEED}" \
  --seq-len 512 \
  --max-series-per-split 500 \
  --max-windows-per-series 4 \
  --layers 3 6 9 11 \
  --token-positions all_context \
  --pooling-modes series_mean \
  --dtype fp32

echo "moment random activation seed=${SEED} end=$(date)"
