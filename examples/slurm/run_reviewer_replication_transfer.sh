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
# Five-resplit zero-shot transfer of BOOM-trained dynamic probes.
#SBATCH --job-name=toto-xfer5
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/reviewer_rep_transfer_%A_%a.out
#SBATCH --error=logs/reviewer_rep_transfer_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_evalmode_20260728}"
LSF_PATH="${LSF_PATH:-${REPO}/data/lsf_datasets}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"

PROBE_ROOT="${RUNS_ROOT}/seed_${SEED}/toto_dynamic_probes"
OUTPUT_ROOT="${RUNS_ROOT}/seed_${SEED}/transfer"

echo "transfer replication seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" "${REPO}/scripts/run_toto_transfer.py" \
  --probe-dir "${PROBE_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --dataset both \
  --fev-safe-only \
  --max-series 100 \
  --max-windows-per-series 4 \
  --lsf-path "${LSF_PATH}" \
  --lsf-datasets ETTh1 ETTh2 weather electricity \
  --context-length 1024 \
  --device cuda \
  --validation-select-one-per-label \
  --require-eval-provenance

echo "transfer replication seed=${SEED} end=$(date)"
