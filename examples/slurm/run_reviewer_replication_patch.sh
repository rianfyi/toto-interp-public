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
# Five seeded, series-disjoint BOOM resplits for the matched future-burstiness patch.
# The fixed probe view is fit separately within each resplit.
#SBATCH --job-name=toto-patch5
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=42-46%5
#SBATCH --output=logs/reviewer_rep_patch_%A_%a.out
#SBATCH --error=logs/reviewer_rep_patch_%A_%a.err

set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO}/runs/reviewer3_evalmode_20260728}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs/reviewer_replications_evalmode_20260728}"
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

SEED_ROOT="${RUNS_ROOT}/seed_${SEED}"
ACT_ROOT="${SOURCE_ROOT}/seed_${SEED}/pretrained_activations"
PROBE_ROOT="${SEED_ROOT}/future_burstiness_probe"
mkdir -p "${REPO}/logs" "${SEED_ROOT}"

echo "patch replication seed=${SEED} host=$(hostname) start=$(date)"

"${PYTHON}" - "${ACT_ROOT}/activation_dump_summary.json" "${SEED}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
assert int(summary["seed"]) == seed, (summary["seed"], seed)
assert summary["model_training"] is False
assert int(summary["context_length"]) == 1024
assert int(summary["max_series_per_split"]) == 500
assert not any(int(value) for value in summary["split_overlap_counts"].values())
assert set(summary["series_id_sha256_by_split"]) == {"train", "val", "test"}
PY

"${PYTHON}" "${REPO}/scripts/fit_toto_probes.py" \
  --activation-files \
    "${ACT_ROOT}/train_activations.pt" \
    "${ACT_ROOT}/val_activations.pt" \
    "${ACT_ROOT}/test_activations.pt" \
  --output-dir "${PROBE_ROOT}" \
  --method linear_probe \
  --label-group dynamic \
  --labels future_burstiness \
  --layers 11 \
  --token-positions all_context \
  --pooling-modes series_mean \
  --seed "${SEED}" \
  --n-jobs 1 \
  --reuse-artifacts

PROBE_PATH="${PROBE_ROOT}/artifacts/future_burstiness__layer_11__all_context__series_mean.pt"
test -f "${PROBE_PATH}"

for BLEND in 0.25 0.5 1.0; do
  BLEND_TAG="${BLEND//./p}"
  OUTPUT_DIR="${SEED_ROOT}/paired_patch_blend_${BLEND_TAG}"
  if [[ -s "${OUTPUT_DIR}/paired_patch_results.csv" && -s "${OUTPUT_DIR}/paired_patch_meta.json" ]]; then
    echo "paired patch already complete seed=${SEED} blend=${BLEND}"
    continue
  fi
  "${PYTHON}" "${REPO}/scripts/run_toto_paired_patch.py" \
    --probe-path "${PROBE_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --snapshot-path "${SNAPSHOT_PATH}" \
    --device cuda \
    --split test \
    --split-seed "${SEED}" \
    --seeds "${SEED}" \
    --context-length 1024 \
    --max-series 500 \
    --max-windows-per-series 4 \
    --max-windows-eval 2000 \
    --num-pairs 40 \
    --num-samples 16 \
    --blend "${BLEND}"
done

echo "patch replication seed=${SEED} end=$(date)"
