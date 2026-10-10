#!/bin/bash
# Run directly on a networked login node, not via sbatch:
#   REPO=/abs/path/to/checkout SOFTWARE_STACK=<module> bash examples/slurm/setup_cluster.sh
# VENV defaults to ${REPO}/.venv-gpu. Other ${VAR:-default} values below are
# functional defaults.
# Create an isolated GPU-capable environment and stage all assets while the
# login node still has network access.  Do not run this on a compute node.
set -euo pipefail

REPO="${REPO:-${SCRATCH}}"
VENV="${VENV:-${REPO}/.venv-gpu}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-${REPO}/data/boom_snapshot}"

module load ${SOFTWARE_STACK}
if [ ! -x "${VENV}/bin/python" ]; then
  python -m venv "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install -e "${REPO}"
"${VENV}/bin/python" -m pip check

export HF_HOME="${REPO}/.cache/huggingface"
"${VENV}/bin/python" "${REPO}/scripts/stage_assets.py" \
  --snapshot-path "${SNAPSHOT_PATH}" \
  --seeds 42 43 44 45 46 \
  --max-series-per-split 500
