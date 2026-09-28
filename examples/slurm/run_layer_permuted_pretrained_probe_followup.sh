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
# CPU-only probe fitting for the five completed layer-permuted activation dumps.
# Submit with, for example:
#   sbatch --dependency=aftercorr:<A10_ARRAY_ID> examples/slurm/run_layer_permuted_pretrained_probe_followup.sh
#SBATCH --job-name=toto-layerperm-probes
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/layerperm_probe_%A_%a.out
#SBATCH --error=logs/layerperm_probe_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/layer_permuted_pretrained}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

ACT_ROOT="${RUNS_ROOT}/seed_${SEED}/layer_permuted_pretrained_activations"
PROBE_ROOT="${RUNS_ROOT}/seed_${SEED}/layer_permuted_pretrained_probes"
for split in train val test; do
  test -s "${ACT_ROOT}/${split}_activations.pt"
done
mkdir -p "${REPO}/logs" "${PROBE_ROOT}"

echo "layer-permuted probes seed=${SEED} host=$(hostname) start=$(date)"
"${PYTHON}" "${REPO}/scripts/fit_toto_probes.py" \
  --activation-files \
    "${ACT_ROOT}/train_activations.pt" \
    "${ACT_ROOT}/val_activations.pt" \
    "${ACT_ROOT}/test_activations.pt" \
  --window-files \
    "${ACT_ROOT}/train_windows.pt" \
    "${ACT_ROOT}/val_windows.pt" \
    "${ACT_ROOT}/test_windows.pt" \
  --output-dir "${PROBE_ROOT}" \
  --method linear_probe \
  --label-group taxonomy \
  --seed "${SEED}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --reuse-artifacts

echo "layer-permuted probes seed=${SEED} end=$(date)"
