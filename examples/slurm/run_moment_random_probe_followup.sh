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
# CPU-only MOMENT random-init probes and provenance. For a normal run, submit
# with --dependency=aftercorr:<RANDOM_ACTIVATION_ARRAY_ID>. For recovery from a
# canceled parent whose files were independently verified, submit only those
# verified seed indices directly. An array-wide afternotok dependency can
# strand completed seeds, while a genuinely failed activation task cannot
# satisfy this script's file precondition.
#SBATCH --job-name=moment-rand-probes
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/moment_random_probe_followup_%A_%a.out
#SBATCH --error=logs/moment_random_probe_followup_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_20260728}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
PYTHON="${VENV}/bin/python"

module load ${SOFTWARE_STACK}
test -x "${PYTHON}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

SEED_ROOT="${RUNS_ROOT}/seed_${SEED}/moment_random"
ACT_ROOT="${SEED_ROOT}/activations"
for split in train val test; do
  test -s "${ACT_ROOT}/${split}_activations.pt"
done

for LABEL_GROUP in taxonomy dynamic; do
  "${PYTHON}" "${REPO}/scripts/fit_toto_probes.py" \
    --activation-files \
      "${ACT_ROOT}/train_activations.pt" \
      "${ACT_ROOT}/val_activations.pt" \
      "${ACT_ROOT}/test_activations.pt" \
    --output-dir "${SEED_ROOT}/${LABEL_GROUP}_probes" \
    --label-group "${LABEL_GROUP}" \
    --method linear_probe \
    --seed "${SEED}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --reuse-artifacts
done

"${PYTHON}" "${REPO}/scripts/enrich_moment_run_provenance.py" \
  --seed-root "${SEED_ROOT}" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --seed "${SEED}" \
  --weight-source random_init
