#!/usr/bin/env bash
# Run the paper's pipeline on one machine, stage by stage, for each seed.
#
# Each stage runs the same Python commands, settings and output folders as the
# Slurm launchers in examples/slurm/ that produced the reported results. A loop
# over --seeds replaces the Slurm array, and running the stages in order
# replaces the job dependencies. Stage data and models first (docs/reproducing.md).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_pipeline.sh [--stages "1 2 ..."] [--seeds "42 ..."] [--dry-run]

Runs the pipeline on one machine. Stage the data and models first
(docs/reproducing.md, "Data and models").

Options:
  --stages LIST   stages to run, e.g. "1 2" (default: all; always run in order)
  --seeds LIST    resplit seeds, e.g. "42" (default: 42 43 44 45 46)
  --dry-run       print every command without running it
  -h, --help      show this help

Stages:
  1  Toto activations, Cramer's V, raw-window models         GPU
  2  Toto taxonomy linear probes                              CPU
  3  Held-out-combination tests, common-support probes        CPU
  4  Block-permuted Toto activations, then probes             GPU, then CPU
  5  MOMENT activations (pretrained, random-init), then       GPU, then CPU
     taxonomy/dynamic probes and provenance enrichment
  6  Toto donor exchange                                      GPU
  7  MOMENT matched interchange                               GPU
  8  Toto dynamic probes, then zero-shot transfer             CPU, then GPU
  9  Summarize, audit, and verify (always seeds 42-46)        CPU

Environment:
  PYTHON    interpreter (default: .venv-gpu/bin/python if it exists, else python3)
  RUNS      parent of the per-stage output folders (default: runs)
  HF_HOME   Hugging Face cache (default: <repo>/.cache/huggingface)
  THREADS   parallel probe fits per run (default: 16, as in the reported runs;
            lower it to reduce peak memory)
  REUSE_EXISTING=1  skip finished steps in stage 1 and finished block-permuted
                    dumps in stage 4
EOF
}

die() { echo "run_pipeline.sh: $*" >&2; exit 1; }

STAGES="1 2 3 4 5 6 7 8 9"
SEEDS="42 43 44 45 46"
DRY_RUN=0
while (($#)); do
  case $1 in
    --stages) [[ $# -ge 2 ]] || die "--stages needs a value"; STAGES=$2; shift 2 ;;
    --seeds) [[ $# -ge 2 ]] || die "--seeds needs a value"; SEEDS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done
[[ -n ${STAGES// /} ]] || die "--stages is empty"
[[ -n ${SEEDS// /} ]] || die "--seeds is empty"
for s in $STAGES; do [[ $s =~ ^[1-9]$ ]] || die "unknown stage: $s (stages are 1-9)"; done
for s in $SEEDS; do [[ $s =~ ^[1-9][0-9]*$ ]] || die "seeds must be positive integers without leading zeros, got: $s"; done

# Stages 1-8 pass absolute paths, as the launchers do. Relative PYTHON, RUNS
# and HF_HOME values are taken from the current directory. Everything then
# runs from the repository root.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x $REPO/.venv-gpu/bin/python ]]; then PYTHON=$REPO/.venv-gpu/bin/python; else PYTHON=python3; fi
elif [[ $PYTHON == */* && $PYTHON != /* ]]; then
  PYTHON=$PWD/$PYTHON
fi
RUNS=${RUNS:-$REPO/runs}
[[ $RUNS == /* ]] || RUNS=$PWD/$RUNS
HF_HOME=${HF_HOME:-$REPO/.cache/huggingface}
[[ $HF_HOME == /* ]] || HF_HOME=$PWD/$HF_HOME
# The launchers passed --n-jobs 16 (--cpus-per-task=16). Fits are seeded, so
# this changes only speed and peak memory, not results.
THREADS=${THREADS:-16}
SNAPSHOT_PATH=$REPO/data/boom_snapshot
LSF_PATH=$REPO/data/lsf_datasets
cd "$REPO"

# ---------------------------------------------------------------- helpers

# Print a command on one line, quoting only the arguments that need it.
render() {
  local arg plain='^[A-Za-z0-9_./:=@%+,-]+$' out=()
  for arg in "$@"; do
    if [[ $arg =~ $plain ]]; then out+=("$arg")
    elif [[ $arg != *"'"* ]]; then out+=("'$arg'")
    else out+=("$(printf '%q' "$arg")"); fi
  done
  echo "${out[*]}"
}

# Echo a command with a "+ " prefix, then run it (unless --dry-run).
run() {
  echo "+ $(render "$@")"
  if [[ $DRY_RUN == 0 ]]; then "$@"; fi
}

# Export NAME=VALUE and echo it like a command.
setenv() {
  echo "+ export $1"
  export "${1?}"
}

# The launchers' `test -s` checks that an earlier stage's output exists.
need_file() {
  run test -s "$1" || die "missing $1 (run the earlier stages for seed $2 first)"
}

GPU_CHECKED=0
need_gpu() {
  if [[ $DRY_RUN == 1 || $GPU_CHECKED == 1 ]]; then return 0; fi
  "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "stage $1 needs a CUDA GPU, but $PYTHON cannot see one (torch.cuda.is_available() failed or is False)"
  GPU_CHECKED=1
}

stage_header() { echo; echo "==> Stage $1: $2"; }
seed_line() { echo "-- seed $2 ($1)"; }

# Environment of the block-permutation, MOMENT, donor-exchange and
# interchange GPU launchers.
gpu_offline_env() {
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  setenv HUGGING_FACE_HUB_OFFLINE=1
  setenv HF_DATASETS_OFFLINE=1
  setenv TRANSFORMERS_OFFLINE=1
}

# The same checks as the inline Python block in run_toto_donor_exchange.sh.
DONOR_SUMMARY_CHECK='import json, sys; s = json.load(open(sys.argv[1])); seed = int(sys.argv[2]); assert int(s["seed"]) == seed, (s["seed"], seed); assert s["model_training"] is False; assert int(s["context_length"]) == 1024; assert int(s["max_series_per_split"]) == 500; assert not any(int(v) for v in s["split_overlap_counts"].values()); assert set(s["series_id_sha256_by_split"]) == {"train", "val", "test"}'

# Each launcher below runs in a subshell so its exports stay local to it.

# ---------------------------------------------------------------- stage 1
# examples/slurm/run_taxonomy_suite.sh
taxonomy_suite() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  setenv HUGGING_FACE_HUB_OFFLINE=1
  setenv HF_DATASETS_OFFLINE=1
  run mkdir -p "$RUNS/taxonomy_suite"
  for SEED in $SEEDS; do
    seed_line run_taxonomy_suite.sh "$SEED"
    run "$PYTHON" "$REPO/scripts/run_taxonomy_suite.py" \
      --output-root "$RUNS/taxonomy_suite" \
      --seed "$SEED" \
      --device cuda \
      --snapshot-path "$SNAPSHOT_PATH" \
      --context-length 1024 \
      --max-series-per-split 500 \
      --max-windows-per-series 4 \
      --raw-width 32 \
      --raw-layers 3 \
      --raw-modes 16 \
      --epochs 20 \
      --batch-size 16 \
      --structural-holdout-target frequency_bucket \
      --structural-holdout-mode combination \
      --structural-holdout-rotation "$((SEED - 42))" \
      --raw-only \
      ${REUSE_EXISTING:+--reuse-existing}
  done
)

# ---------------------------------------------------------------- stage 2
# examples/slurm/run_taxonomy_probes.sh
taxonomy_probes() (
  for SEED in $SEEDS; do
    seed_line run_taxonomy_probes.sh "$SEED"
    run "$PYTHON" "$REPO/scripts/run_taxonomy_probes.py" \
      --runs-root "$RUNS/taxonomy_suite" \
      --seed "$SEED" \
      --n-jobs "$THREADS" \
      --raw-width 32 \
      --raw-layers 3 \
      --raw-modes 16 \
      --epochs 20 \
      --batch-size 16 \
      --structural-holdout-target frequency_bucket \
      --structural-holdout-mode combination \
      --structural-holdout-rotation "$((SEED - 42))" \
      --skip-post-controls \
      --reuse-existing
  done
)

# ---------------------------------------------------------------- stage 3
# examples/slurm/run_holdout_grid.sh
holdout_grid() (
  for SEED in $SEEDS; do
    seed_line run_holdout_grid.sh "$SEED"
    SEED_ROOT=$RUNS/taxonomy_suite/seed_$SEED
    for SOURCE in pretrained random_init; do
      run "$PYTHON" "$REPO/scripts/run_holdout_grid.py" \
        --activation-files \
          "$SEED_ROOT/${SOURCE}_activations/train_activations.pt" \
          "$SEED_ROOT/${SOURCE}_activations/val_activations.pt" \
          "$SEED_ROOT/${SOURCE}_activations/test_activations.pt" \
        --output-root "$SEED_ROOT/structural_holdout" \
        --conditional-output-dir "$SEED_ROOT/conditional/$SOURCE" \
        --source "$SOURCE" \
        --seed "$SEED" \
        --rotation "$((SEED - 42))" \
        --reuse-existing
    done
  done
)

# ---------------------------------------------------------------- stage 4
# examples/slurm/run_layer_permuted_pretrained_a10.sh
layer_permuted_dump() (
  gpu_offline_env
  for SEED in $SEEDS; do
    seed_line run_layer_permuted_pretrained_a10.sh "$SEED"
    ACT_ROOT=$RUNS/layer_permuted_pretrained/seed_$SEED/layer_permuted_pretrained_activations
    run mkdir -p "$ACT_ROOT"
    # As in the launcher, dump again unless REUSE_EXISTING=1 and a finished dump exists.
    if [[ -s $ACT_ROOT/activation_dump_summary.json && ${REUSE_EXISTING:-0} == 1 ]]; then
      echo "   activations already dumped, skipping"
      continue
    fi
    # The same seed sets the BOOM resplit and the block permutation.
    run "$PYTHON" "$REPO/scripts/dump_toto_activations.py" \
      --output-dir "$ACT_ROOT" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --device cuda \
      --seed "$SEED" \
      --layer-permutation-seed "$SEED" \
      --weight-source layer_permuted_pretrained \
      --context-length 1024 \
      --max-series-per-split 500 \
      --max-windows-per-series 4 \
      --token-positions all_context final_context first_decode \
      --pooling-modes per_variate series_mean
  done
)

# examples/slurm/run_layer_permuted_pretrained_probe_followup.sh
layer_permuted_probes() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  for SEED in $SEEDS; do
    seed_line run_layer_permuted_pretrained_probe_followup.sh "$SEED"
    ACT_ROOT=$RUNS/layer_permuted_pretrained/seed_$SEED/layer_permuted_pretrained_activations
    PROBE_ROOT=$RUNS/layer_permuted_pretrained/seed_$SEED/layer_permuted_pretrained_probes
    for split in train val test; do need_file "$ACT_ROOT/${split}_activations.pt" "$SEED"; done
    run mkdir -p "$PROBE_ROOT"
    run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
      --activation-files \
        "$ACT_ROOT/train_activations.pt" \
        "$ACT_ROOT/val_activations.pt" \
        "$ACT_ROOT/test_activations.pt" \
      --window-files \
        "$ACT_ROOT/train_windows.pt" \
        "$ACT_ROOT/val_windows.pt" \
        "$ACT_ROOT/test_windows.pt" \
      --output-dir "$PROBE_ROOT" \
      --method linear_probe \
      --label-group taxonomy \
      --seed "$SEED" \
      --n-jobs "$THREADS" \
      --reuse-artifacts
  done
)

# ---------------------------------------------------------------- stage 5
# examples/slurm/run_moment.sh (pretrained MOMENT-base)
moment_dump() (
  gpu_offline_env
  for SEED in $SEEDS; do
    seed_line run_moment.sh "$SEED"
    SEED_ROOT=$RUNS/moment_suite/seed_$SEED/moment
    ACT_ROOT=$SEED_ROOT/activations
    run mkdir -p "$SEED_ROOT"
    if [[ -s $ACT_ROOT/activation_dump_summary.json ]]; then
      echo "   activations already dumped, skipping"
      continue
    fi
    run "$PYTHON" "$REPO/scripts/dump_moment_activations.py" \
      --output-dir "$ACT_ROOT" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --device cuda \
      --seed "$SEED" \
      --seq-len 512 \
      --max-series-per-split 500 \
      --max-windows-per-series 4 \
      --layers 3 6 9 11 \
      --token-positions all_context \
      --pooling-modes series_mean \
      --dtype fp32
  done
)

# examples/slurm/run_moment_random.sh (random-init MOMENT-base)
moment_random_dump() (
  gpu_offline_env
  for SEED in $SEEDS; do
    seed_line run_moment_random.sh "$SEED"
    SEED_ROOT=$RUNS/moment_suite/seed_$SEED/moment_random
    run mkdir -p "$SEED_ROOT"
    run "$PYTHON" "$REPO/scripts/dump_moment_activations.py" \
      --output-dir "$SEED_ROOT/activations" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --device cuda \
      --weight-source random_init \
      --seed "$SEED" \
      --seq-len 512 \
      --max-series-per-split 500 \
      --max-windows-per-series 4 \
      --layers 3 6 9 11 \
      --token-positions all_context \
      --pooling-modes series_mean \
      --dtype fp32
  done
)

# examples/slurm/run_moment_dynamic.sh (pretrained taxonomy and dynamic probes)
moment_probes() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  for SEED in $SEEDS; do
    seed_line run_moment_dynamic.sh "$SEED"
    SEED_ROOT=$RUNS/moment_suite/seed_$SEED/moment
    ACT_ROOT=$SEED_ROOT/activations
    for split in train val test; do need_file "$ACT_ROOT/${split}_activations.pt" "$SEED"; done
    # Taxonomy probes also read the window files; dynamic probes do not.
    run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
      --activation-files \
        "$ACT_ROOT/train_activations.pt" \
        "$ACT_ROOT/val_activations.pt" \
        "$ACT_ROOT/test_activations.pt" \
      --window-files \
        "$ACT_ROOT/train_windows.pt" \
        "$ACT_ROOT/val_windows.pt" \
        "$ACT_ROOT/test_windows.pt" \
      --output-dir "$SEED_ROOT/probes" \
      --label-group taxonomy \
      --method linear_probe \
      --seed "$SEED" \
      --n-jobs "$THREADS" \
      --reuse-artifacts
    run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
      --activation-files \
        "$ACT_ROOT/train_activations.pt" \
        "$ACT_ROOT/val_activations.pt" \
        "$ACT_ROOT/test_activations.pt" \
      --output-dir "$SEED_ROOT/dynamic_probes" \
      --label-group dynamic \
      --method linear_probe \
      --seed "$SEED" \
      --n-jobs "$THREADS" \
      --reuse-artifacts
  done
)

# examples/slurm/run_moment_random_probe_followup.sh (random-init probes and provenance)
moment_random_probes() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  for SEED in $SEEDS; do
    seed_line run_moment_random_probe_followup.sh "$SEED"
    SEED_ROOT=$RUNS/moment_suite/seed_$SEED/moment_random
    ACT_ROOT=$SEED_ROOT/activations
    for split in train val test; do need_file "$ACT_ROOT/${split}_activations.pt" "$SEED"; done
    for LABEL_GROUP in taxonomy dynamic; do
      run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
        --activation-files \
          "$ACT_ROOT/train_activations.pt" \
          "$ACT_ROOT/val_activations.pt" \
          "$ACT_ROOT/test_activations.pt" \
        --output-dir "$SEED_ROOT/${LABEL_GROUP}_probes" \
        --label-group "$LABEL_GROUP" \
        --method linear_probe \
        --seed "$SEED" \
        --n-jobs "$THREADS" \
        --reuse-artifacts
    done
    run "$PYTHON" "$REPO/scripts/enrich_moment_run_provenance.py" \
      --seed-root "$SEED_ROOT" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --seed "$SEED" \
      --weight-source random_init
  done
)

# ---------------------------------------------------------------- stage 6
# examples/slurm/run_toto_donor_exchange.sh
toto_donor_exchange() (
  gpu_offline_env
  for SEED in $SEEDS; do
    seed_line run_toto_donor_exchange.sh "$SEED"
    SEED_ROOT=$RUNS/toto_exchange_transfer/seed_$SEED
    ACT_ROOT=$RUNS/taxonomy_suite/seed_$SEED/pretrained_activations
    PROBE_ROOT=$SEED_ROOT/future_burstiness_probe
    run mkdir -p "$SEED_ROOT"
    # Check that stage 1's activation dump has the expected settings.
    run "$PYTHON" -c "$DONOR_SUMMARY_CHECK" "$ACT_ROOT/activation_dump_summary.json" "$SEED"
    # Fit the layer-11, all-context, series-mean future-burstiness probe.
    run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
      --activation-files \
        "$ACT_ROOT/train_activations.pt" \
        "$ACT_ROOT/val_activations.pt" \
        "$ACT_ROOT/test_activations.pt" \
      --output-dir "$PROBE_ROOT" \
      --method linear_probe \
      --label-group dynamic \
      --labels future_burstiness \
      --layers 11 \
      --token-positions all_context \
      --pooling-modes series_mean \
      --seed "$SEED" \
      --n-jobs 1 \
      --reuse-artifacts
    PROBE_PATH=$PROBE_ROOT/artifacts/future_burstiness__layer_11__all_context__series_mean.pt
    run test -f "$PROBE_PATH" || die "missing $PROBE_PATH"
    for BLEND in 0.25 0.5 1.0; do
      OUTPUT_DIR=$SEED_ROOT/paired_patch_blend_${BLEND//./p}
      if [[ -s $OUTPUT_DIR/paired_patch_results.csv && -s $OUTPUT_DIR/paired_patch_meta.json ]]; then
        echo "   blend $BLEND already complete, skipping"
        continue
      fi
      run "$PYTHON" "$REPO/scripts/run_toto_paired_patch.py" \
        --probe-path "$PROBE_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --snapshot-path "$SNAPSHOT_PATH" \
        --device cuda \
        --split test \
        --split-seed "$SEED" \
        --seeds "$SEED" \
        --context-length 1024 \
        --max-series 500 \
        --max-windows-per-series 4 \
        --max-windows-eval 2000 \
        --num-pairs 40 \
        --num-samples 16 \
        --blend "$BLEND"
    done
  done
)

# ---------------------------------------------------------------- stage 7
# examples/slurm/run_moment_interchange.sh
moment_interchange() (
  gpu_offline_env
  for SEED in $SEEDS; do
    seed_line run_moment_interchange.sh "$SEED"
    SEED_ROOT=$RUNS/moment_suite/seed_$SEED/moment
    ACTIVATION_SUMMARY=$SEED_ROOT/activations/activation_dump_summary.json
    PROBE_RESULTS=$SEED_ROOT/dynamic_probes/probe_results.csv
    OUTPUT_DIR=$SEED_ROOT/interchange
    need_file "$ACTIVATION_SUMMARY" "$SEED"
    need_file "$PROBE_RESULTS" "$SEED"
    run mkdir -p "$OUTPUT_DIR"
    # Provenance enrichment for the pretrained MOMENT run.
    run "$PYTHON" "$REPO/scripts/enrich_moment_run_provenance.py" \
      --seed-root "$SEED_ROOT" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --seed "$SEED"
    run "$PYTHON" "$REPO/scripts/run_moment_paired_interchange.py" \
      --probe-results "$PROBE_RESULTS" \
      --activation-summary "$ACTIVATION_SUMMARY" \
      --output-dir "$OUTPUT_DIR" \
      --snapshot-path "$SNAPSHOT_PATH" \
      --model-id AutonLab/MOMENT-1-base \
      --split-seed "$SEED" \
      --sampling-seed "$SEED" \
      --device cuda \
      --dtype fp32 \
      --num-pairs 40 \
      --blends 0.25 0.5 1.0 \
      --high-quantile 0.75 \
      --low-quantile 0.25 \
      --null-match-k 5 \
      --secondary-endpoint auto
  done
)

# ---------------------------------------------------------------- stage 8
# examples/slurm/run_toto_dynamic.sh
toto_dynamic() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  for SEED in $SEEDS; do
    seed_line run_toto_dynamic.sh "$SEED"
    ACT_ROOT=$RUNS/taxonomy_suite/seed_$SEED/pretrained_activations
    run "$PYTHON" "$REPO/scripts/fit_toto_probes.py" \
      --activation-files \
        "$ACT_ROOT/train_activations.pt" \
        "$ACT_ROOT/val_activations.pt" \
        "$ACT_ROOT/test_activations.pt" \
      --output-dir "$RUNS/toto_exchange_transfer/seed_$SEED/toto_dynamic_probes" \
      --label-group dynamic \
      --method linear_probe \
      --seed "$SEED" \
      --n-jobs "$THREADS" \
      --reuse-artifacts
  done
)

# examples/slurm/run_transfer.sh (this launcher sets no offline flags)
transfer() (
  setenv "PYTHONPATH=$REPO:${PYTHONPATH:-}"
  for SEED in $SEEDS; do
    seed_line run_transfer.sh "$SEED"
    run "$PYTHON" "$REPO/scripts/run_toto_transfer.py" \
      --probe-dir "$RUNS/toto_exchange_transfer/seed_$SEED/toto_dynamic_probes" \
      --output-dir "$RUNS/toto_exchange_transfer/seed_$SEED/transfer" \
      --dataset both \
      --fev-safe-only \
      --max-series 100 \
      --max-windows-per-series 4 \
      --lsf-path "$LSF_PATH" \
      --lsf-datasets ETTh1 ETTh2 weather electricity \
      --context-length 1024 \
      --device cuda \
      --validation-select-one-per-label \
      --require-eval-provenance
  done
)

# ---------------------------------------------------------------- stage 9
# The summarize, audit and verify commands from docs/reproducing.md. The
# summaries and the audit always cover the paper's five seeds (42-46). As
# documented there, this
# stage uses paths relative to the repository root (runs/...) when RUNS is
# inside it; the taxonomy summaries then store bare artifact file names, as
# in results/toto_taxonomy/.
summarize_audit_verify() (
  R=${RUNS#"$REPO"/}
  if [[ $PYTHON == */* ]]; then
    # verify_against.sh calls python3; use the same environment as $PYTHON.
    echo "+ export PATH=$(dirname "$PYTHON"):\$PATH"
    PATH=$(dirname "$PYTHON"):$PATH
    export PATH
  fi
  run "$PYTHON" scripts/summarize_taxonomy_suite.py \
    --runs-root "$R/taxonomy_suite" \
    --layer-permuted-runs-root "$R/layer_permuted_pretrained" \
    --output-dir "$R/summary/toto_taxonomy"
  run "$PYTHON" scripts/summarize_moment_exchange_transfer.py \
    --runs-root "$R/toto_exchange_transfer" \
    --moment-runs-root "$R/moment_suite" \
    --output-dir "$R/summary/moment_exchange_transfer" \
    --seeds 42 43 44 45 46
  run "$PYTHON" scripts/audit_results.py \
    --moment-exchange-transfer-summary-dir "$R/summary/moment_exchange_transfer" \
    --taxonomy-summary-dir "$R/summary/toto_taxonomy" \
    --output "$R/summary/E2E_AUDIT.json"
  run bash reproduce/verify_against.sh "$R/summary"
)

# ---------------------------------------------------------------- main

if [[ $DRY_RUN == 0 ]]; then
  command -v "$PYTHON" >/dev/null || die "Python not found: $PYTHON (set PYTHON=...)"
fi
echo "Stages: $STAGES   Seeds: $SEEDS"
echo "PYTHON=$PYTHON  RUNS=$RUNS  THREADS=$THREADS"
if [[ $DRY_RUN == 1 ]]; then echo "Dry run: printing commands only."; fi
setenv "HF_HOME=$HF_HOME"

for n in 1 2 3 4 5 6 7 8 9; do
  [[ " $STAGES " == *" $n "* ]] || continue
  case $n in
    1) stage_header 1 "Toto activations, Cramer's V, raw-window models (GPU)"
       need_gpu 1
       taxonomy_suite ;;
    2) stage_header 2 "Toto taxonomy linear probes (CPU)"
       taxonomy_probes ;;
    3) stage_header 3 "Held-out-combination tests, common-support probes (CPU)"
       holdout_grid ;;
    4) stage_header 4 "Block-permuted Toto activations (GPU), then probes (CPU)"
       need_gpu 4
       layer_permuted_dump
       layer_permuted_probes ;;
    5) stage_header 5 "MOMENT activations (GPU), then probes and provenance (CPU)"
       need_gpu 5
       moment_dump
       moment_random_dump
       moment_probes
       moment_random_probes ;;
    6) stage_header 6 "Toto donor exchange (GPU)"
       need_gpu 6
       toto_donor_exchange ;;
    7) stage_header 7 "MOMENT matched interchange (GPU)"
       need_gpu 7
       moment_interchange ;;
    8) stage_header 8 "Toto dynamic probes (CPU), then zero-shot transfer (GPU)"
       need_gpu 8
       toto_dynamic
       transfer ;;
    9) stage_header 9 "Summarize, audit, and verify (CPU)"
       summarize_audit_verify ;;
  esac
done
echo
echo "Done."
