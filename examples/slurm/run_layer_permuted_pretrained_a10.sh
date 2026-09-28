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
# Five seeded pretrained-block-order controls.  This job only performs the
# GPU-bound activation dumps; submit the CPU follow-up after it completes.
#SBATCH --job-name=toto-layerperm
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/layerperm_%A_%a.out
#SBATCH --error=logs/layerperm_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/layer_permuted_pretrained}"
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

ACT_ROOT="${RUNS_ROOT}/seed_${SEED}/layer_permuted_pretrained_activations"
mkdir -p "${REPO}/logs" "${ACT_ROOT}"
echo "layer-permuted Toto seed=${SEED} host=$(hostname) start=$(date)"

# The same seed is used for the BOOM resplit and the explicit permutation.
# activation_dump_summary.json records the exact runtime->pretrained map.
if [[ ! -s "${ACT_ROOT}/activation_dump_summary.json" || "${REUSE_EXISTING:-0}" != "1" ]]; then
  "${PYTHON}" "${REPO}/scripts/dump_toto_activations.py" \
    --output-dir "${ACT_ROOT}" \
    --snapshot-path "${SNAPSHOT_PATH}" \
    --device cuda \
    --seed "${SEED}" \
    --layer-permutation-seed "${SEED}" \
    --weight-source layer_permuted_pretrained \
    --context-length 1024 \
    --max-series-per-split 500 \
    --max-windows-per-series 4 \
    --token-positions all_context final_context first_decode \
    --pooling-modes per_variate series_mean
fi

echo "layer-permuted Toto seed=${SEED} end=$(date)"
