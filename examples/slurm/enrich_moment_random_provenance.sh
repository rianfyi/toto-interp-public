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
# Add derivable eval/split provenance to the completed five-seed MOMENT random control.
# Submit after the random-init array completes.
#SBATCH --job-name=moment-rand-prov
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/moment_random_provenance_%A_%a.out
#SBATCH --error=logs/moment_random_provenance_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_20260728}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"
SEED="${SLURM_ARRAY_TASK_ID:?}"

module load ${SOFTWARE_STACK}
"${VENV}/bin/python" "${REPO}/scripts/enrich_moment_run_provenance.py" \
  --seed-root "${RUNS_ROOT}/seed_${SEED}/moment_random" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --seed "${SEED}" \
  --weight-source random_init
