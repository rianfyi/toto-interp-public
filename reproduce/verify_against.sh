#!/bin/bash
# Run the packaged paper-number verifier against any results directory
# WITHOUT editing reproduce/verify_paper_numbers.py.
#
# The verifier resolves its inputs relative to its own location
# (<pkg>/results + <pkg>/reproduce/expected_numbers.csv), so this wrapper
# builds a staging tree of symlinks and runs the unmodified script there.
#
# Usage:
#   bash reproduce/verify_against.sh results                  # packaged audited results
#   bash reproduce/verify_against.sh /path/to/rerun/results   # a full pipeline rerun
#
# Expected on the packaged results: PASS 424 / FAIL 0. A rerun uses the same
# subfolder layout as results/. CPU-only, finishes in seconds.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash reproduce/verify_against.sh <results-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ABS="$(cd "$1" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ln -s "$REPO_ROOT/reproduce" "$STAGE/reproduce"
ln -s "$RESULTS_ABS" "$STAGE/results"
python3 "$STAGE/reproduce/verify_paper_numbers.py"
