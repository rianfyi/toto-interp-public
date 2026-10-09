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
# Five series-disjoint MOMENT-base matched-interchange resplits.
# Requires the activation dump and dynamic-probe jobs for the same seed.
#SBATCH --job-name=moment-xchg5
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/moment_interchange_%A_%a.out
#SBATCH --error=logs/moment_interchange_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/moment_suite}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"
NUM_PAIRS="${NUM_PAIRS:-40}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
export HUGGING_FACE_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEED_ROOT="${RUNS_ROOT}/seed_${SEED}/moment"
ACTIVATION_SUMMARY="${SEED_ROOT}/activations/activation_dump_summary.json"
PROBE_RESULTS="${SEED_ROOT}/dynamic_probes/probe_results.csv"
OUTPUT_DIR="${SEED_ROOT}/interchange"

test -s "${ACTIVATION_SUMMARY}"
test -s "${PROBE_RESULTS}"
mkdir -p "${REPO}/logs" "${OUTPUT_DIR}"

echo "moment interchange seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" "${REPO}/scripts/enrich_moment_run_provenance.py" \
  --seed-root "${SEED_ROOT}" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --seed "${SEED}"

"${PYTHON}" "${REPO}/scripts/run_moment_paired_interchange.py" \
  --probe-results "${PROBE_RESULTS}" \
  --activation-summary "${ACTIVATION_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --model-id "AutonLab/MOMENT-1-base" \
  --split-seed "${SEED}" \
  --sampling-seed "${SEED}" \
  --device cuda \
  --dtype fp32 \
  --num-pairs "${NUM_PAIRS}" \
  --blends 0.25 0.5 1.0 \
  --high-quantile 0.75 \
  --low-quantile 0.25 \
  --null-match-k 5 \
  --secondary-endpoint auto

echo "moment interchange seed=${SEED} end=$(date)"
