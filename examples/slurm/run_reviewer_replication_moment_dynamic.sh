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
# CPU-only taxonomy and dynamic probes after pretrained MOMENT activation dumps.
# Submit with --dependency=aftercorr:<MOMENT_ACTIVATION_ARRAY_ID>.
#SBATCH --job-name=moment-probes5
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/reviewer_rep_moment_dynamic_%A_%a.out
#SBATCH --error=logs/reviewer_rep_moment_dynamic_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_20260728}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

ACT_ROOT="${RUNS_ROOT}/seed_${SEED}/moment/activations"

echo "moment probes seed=${SEED} host=$(hostname) start=$(date)"

for split in train val test; do
  test -s "${ACT_ROOT}/${split}_activations.pt"
done

for LABEL_GROUP in taxonomy dynamic; do
  if [[ "${LABEL_GROUP}" == "taxonomy" ]]; then
    PROBE_ROOT="${RUNS_ROOT}/seed_${SEED}/moment/probes"
    WINDOW_ARGS=(
      --window-files
      "${ACT_ROOT}/train_windows.pt"
      "${ACT_ROOT}/val_windows.pt"
      "${ACT_ROOT}/test_windows.pt"
    )
  else
    PROBE_ROOT="${RUNS_ROOT}/seed_${SEED}/moment/dynamic_probes"
    WINDOW_ARGS=()
  fi
  "${PYTHON}" "${REPO}/scripts/fit_toto_probes.py" \
    --activation-files \
      "${ACT_ROOT}/train_activations.pt" \
      "${ACT_ROOT}/val_activations.pt" \
      "${ACT_ROOT}/test_activations.pt" \
    "${WINDOW_ARGS[@]}" \
    --output-dir "${PROBE_ROOT}" \
    --label-group "${LABEL_GROUP}" \
    --method linear_probe \
    --seed "${SEED}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --reuse-artifacts
done

echo "moment probes seed=${SEED} end=$(date)"
