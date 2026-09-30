# toto-interp — code for "A Control-First Representation Audit of an Observability Forecasting Foundation Model" (NeurIPS 2026)

Camera-ready code release: [`neurips-2026-v1`](https://github.com/DhyeyMavani2003/toto-interp/releases/tag/neurips-2026-v1).
Use this tag when reproducing or citing the released package. The release
includes a source ZIP and its SHA-256 checksum.

## License and third-party material

The authors' original code, documentation, and aggregate results are available
under the [MIT License](LICENSE), copyright 2026 toto-interp contributors.
Material adapted from Datadog Toto retains its Apache-2.0 terms and attribution;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
This license does not cover third-party datasets, checkpoints, or dependencies.
No new dataset or model weights are released.

## Reproduction scope

This repository reproduces the reported five-resplit results at two levels:

- **Level A (CPU, minutes):** recompute the reported results (424 checks
  over Tables 1–10, the values plotted in Figures 1, 3 and 4, the
  in-text means, half-widths and win counts, and a 48-cell supplementary
  win-count summary) from the
  compact, audited result CSVs in `results/`, and regenerate the figure data.
- **Level B (GPU cluster):** rerun the five-resplit pipeline from scratch
  (seeds 42–46) with the exact scripts, seeds, and settings used.
  Data and model weights are downloaded by the scripts; they are not
  bundled here.

Code version: `toto_interp/`, `examples/slurm/` and the pipeline scripts
are the code that produced the audited results.
`scripts/plot_camera_ready_figures.py` and
`scripts/make_camera_ready_fig_data.py` compute the plotted figure values
from those results. The only functional change is that
`fit_toto_probes.py` defaults `--gbdt-backend` to `hist_gradient_boosting`,
the backend `auto` resolved to in the audited runs (xgboost was not installed). Shell launchers in `examples/slurm/` use `${...}`
placeholders for site values (see the header in each file). No dataset
files, activations, model weights, or manuscript sources are included.

## Layout

- `toto_interp/` — analysis library (probes, labels, controls, transfer).
- `scripts/` — pipeline scripts (dumps, probe fits, interventions,
  transfer, summarizers, audit, figure generation, data staging).
- `examples/slurm/` — Slurm launchers with site placeholders (Level B).
- `results/` — audited compact CSVs + `E2E_AUDIT.json` + `EXECUTION.md`.
- `reproduce/` — `verify_paper_numbers.py` + `expected_numbers.csv`;
  `verify_against.sh` runs the same checks on another results directory.
- `tests/` — unit tests for `toto_interp/` and the pipeline scripts.
- `docs/` — LSTF dataset setup (`lsf_setup.md`) and recorded asset revisions
  (`asset_provenance.md`).
- `requirements.txt` — frozen pins with per-pin provenance; `pyproject.toml`.
- `LICENSE` — MIT License for the authors' original code, documentation, and
  aggregate results.
- `THIRD_PARTY_NOTICES.md` — attribution for material adapted from Datadog Toto
  and notes on upstream terms for dependencies, datasets, and model weights.
- `licenses/` — Apache-2.0 license text and Datadog Toto NOTICE for the adapted
  files (`toto_interp/fev_tasks.py`, `toto_interp/boom.py`).

## Environment setup

Level A needs only `numpy`, `pandas`, `scipy`, `matplotlib`:

    python -m venv .venv-levela
    source .venv-levela/bin/activate
    pip install numpy pandas scipy matplotlib

Level B (cluster, GPU): from the repository root, build a Python 3.12
venv at `.venv-gpu` (the launchers' default `VENV`), install `torch` with
the CUDA build matching your cluster, then:

    python3.12 -m venv .venv-gpu && source .venv-gpu/bin/activate
    pip install -r requirements.txt
    pip install -e .
    pip install --no-deps momentfm==0.1.4   # its own pins have no Python 3.12 wheels

Stage data and checkpoints on a networked login node before submitting
GPU jobs (see `examples/slurm/setup_reviewer3_cluster.sh`). The launchers
run offline and read the Hugging Face cache under the repository root.
Log in to the Hub first (`huggingface-cli login` or `HF_TOKEN`); the BOOM
snapshot has about 2,700 per-series folders and unauthenticated downloads
hit rate limits.

    export HF_HOME="$PWD/.cache/huggingface"
    python scripts/stage_reviewer3_assets.py --snapshot-path data/boom_snapshot --seeds 42 43 44 45 46 --max-series-per-split 500
    python scripts/stage_fev_safe_datasets.py
    python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets
    python -c "from toto_interp.moment_loader import load_moment_with_fallback; load_moment_with_fallback()"  # caches MOMENT-1-base

## Level A

From the package root:

    python reproduce/verify_paper_numbers.py
    # expect: PASS 424 FAIL 0; exit code 0

To check a rerun (Level B) against the same expected values, point the
verifier at its results directory (same subfolder layout as `results/`):

    bash reproduce/verify_against.sh /path/to/rerun/results

Regenerate the figure data (Figures 1, 3 and 4). The figure scripts
expect the layout `runs/rebuttal/...` and write under
`paper/neurips2026/figures/`, so stage symlinks first (no data are copied):

    mkdir -p runs/rebuttal
    ln -s ../../results/reviewer3_controls runs/rebuttal/reviewer3_controls
    ln -s ../../results/reviewer_replications_5seed runs/rebuttal/reviewer_replications_5seed
    python scripts/plot_camera_ready_figures.py
    python scripts/make_camera_ready_fig_data.py
    # writes paper/neurips2026/figures/data/*.csv (plotted values) and
    # paper/neurips2026/figures/tikz/gen/*.tex; the second script asserts
    # every Table 1 mean and half-width before writing.

Byte-compile check: `python -m compileall -q toto_interp scripts reproduce`.
Unit tests (Level B environment): `python -m pytest -q tests`.

## Level B (GPU cluster, seeds 42–46)

Export `REPO` (the absolute path of this checkout; `SCRATCH` is only a
fallback) and `SOFTWARE_STACK`, run `mkdir -p logs` from the repository
root, and submit each launcher with your site's account and partition
(GPU launchers request `--gres=gpu:A10:1`; override it if needed), e.g.
`sbatch --account=<account> --partition=<partition> examples/slurm/run_reviewer3_suite.sh`.
Each array covers seeds 42–46 and consumes the previous stage's outputs;
`RUNS_ROOT` redirects a stage's output root. The launchers are
authoritative; the commands below list their main settings.

1. **Toto dumps, confounding, and raw-window controls**
   (`examples/slurm/run_reviewer3_suite.sh`, GPU):
   `run_reviewer3_suite.py --seed SEED --raw-only` runs
   `dump_toto_activations.py --context-length 1024
   --max-series-per-split 500 --max-windows-per-series 4`
   (pretrained + random-init), `compute_structural_confounding.py`, and
   the FNO/CNN/Transformer/GBDT raw-window controls (`fit_toto_probes.py`;
   width 32 / 3 layers / 16 modes / 20 epochs / batch 16).
2. **Unconditional linear probes**
   (`examples/slurm/run_reviewer3_cpu_followup.sh`, CPU):
   `run_reviewer3_cpu_followup.py --seed SEED --skip-post-controls
   --reuse-existing` fits the taxonomy linear probes for pretrained and
   random-init activations.
3. **Held-out grid + conditional probes**
   (`examples/slurm/run_reviewer3_holdout_grid.sh`, CPU):
   `run_reviewer3_holdout_grid.py --source {pretrained,random_init}
   --seed SEED --rotation SEED-42` (held-out support via
   `run_structural_holdout_probes.py`, common-support probes via
   `run_conditional_probes.py`).
4. **Block-permutation dumps + probes**
   (`run_layer_permuted_pretrained_a10.sh`, GPU, then
   `run_layer_permuted_pretrained_probe_followup.sh`, CPU):
   `dump_toto_activations.py --layer-permutation-seed SEED
   --weight-source layer_permuted_pretrained` (both pooling modes),
   then probe fits. Default root: `runs/layer_permuted_pretrained`.
5. **MOMENT dumps + probes** (`run_reviewer_replication_moment.sh` /
   `run_reviewer_replication_moment_random.sh`, GPU, then
   `run_reviewer_replication_moment_dynamic.sh` /
   `run_moment_random_probe_followup.sh`, CPU):
   `dump_moment_activations.py --seed SEED --seq-len 512
   --max-series-per-split 500 --max-windows-per-series 4
   --layers 3 6 9 11 --token-positions all_context
   --pooling-modes series_mean` (pretrained and random-init), then
   taxonomy + dynamic probe fits. `enrich_moment_run_provenance.py` runs in
   `run_moment_random_probe_followup.sh` (random-init) and in the step-7
   interchange launcher (pretrained).
   Default root: `runs/reviewer_replications_20260728`.
6. **Toto donor exchange** (`run_reviewer_replication_patch.sh`, GPU):
   fit layer-11/all-context/series-mean future-burstiness probe, then
   `run_toto_paired_patch.py --split test --split-seed SEED --seeds SEED
   --context-length 1024 --max-series 500 --max-windows-per-series 4
   --max-windows-eval 2000 --num-pairs 40 --num-samples 16
   --blend {0.25,0.5,1.0}`.
   Default root: `runs/reviewer_replications_evalmode_20260728`.
7. **MOMENT interchange** (`run_reviewer_replication_moment_interchange.sh`,
   GPU): `run_moment_paired_interchange.py --split-seed SEED
   --sampling-seed SEED --num-pairs 40 --blends 0.25 0.5 1.0
   --high-quantile 0.75 --low-quantile 0.25 --null-match-k 5
   --secondary-endpoint auto`.
8. **Transfer**: first `run_reviewer_replication_toto_dynamic.sh` (CPU;
   fits the Toto dynamic probes that transfer applies), then
   `run_reviewer_replication_transfer.sh` (GPU):
   `run_toto_transfer.py --dataset both --fev-safe-only --max-series 100
   --max-windows-per-series 4 --lsf-datasets ETTh1 ETTh2 weather electricity
   --context-length 1024 --validation-select-one-per-label
   --require-eval-provenance`.
9. **Summaries + audit** (CPU/login node), with the launchers' default roots:

       python scripts/summarize_reviewer3_suite.py \
         --runs-root runs/reviewer3_evalmode_20260728 \
         --layer-permuted-runs-root runs/layer_permuted_pretrained \
         --output-dir runs/summary/reviewer3_controls
       python scripts/summarize_reviewer_replications.py \
         --runs-root runs/reviewer_replications_evalmode_20260728 \
         --moment-runs-root runs/reviewer_replications_20260728 \
         --output-dir runs/summary/reviewer_replications_5seed \
         --seeds 42 43 44 45 46
       python scripts/audit_rebuttal_e2e.py \
         --replication-summary-dir runs/summary/reviewer_replications_5seed \
         --reviewer3-summary-dir runs/summary/reviewer3_controls \
         --output runs/summary/E2E_AUDIT.json
       bash reproduce/verify_against.sh runs/summary

   Expect status complete for all cells (compare `results/E2E_AUDIT.json`).
   Outputs go to `runs/summary/`, not the Level A staging links. Expect small
   differences from the packaged results: in the authors' full rerun of this
   code, 346 of 424 checks matched at printed precision. GPU nondeterminism
   moved the others by a median of about 0.002 (at most about 0.024), flipped
   one near-tie win count (domain vs. strongest raw-window model, 2 to 3 of 5),
   and swapped the near-tied strongest cardinality raw-window family (FNO/CNN).
10. **Figures**: the Level A figure-data commands regenerate the figures
    from the packaged results. Both figure scripts assert the paper's printed
    values, so they stop on rerun outputs that differ.

## Paper table/figure → script → results file

| Paper item | Pipeline stage(s) | Results file(s) |
|---|---|---|
| Table 1 (Toto taxonomy readouts and controls) | `run_reviewer3_suite.py`, `run_reviewer3_cpu_followup.py`, `fit_toto_probes.py`; block permutation: step 4 | `reviewer3_controls/unconditional_selected_all_seeds.csv`, `raw_control_all_seeds.csv`, `layer_permuted_selected_all_seeds.csv` |
| Table 2 (common-support probes) | `run_conditional_probes.py` | `reviewer3_controls/conditional_all_seeds.csv` |
| Table 3 (MOMENT-base taxonomy) | `dump_moment_activations.py`, `fit_toto_probes.py` | `reviewer_replications_5seed/moment_per_resplit.csv`, `moment_random_structural_per_resplit.csv` |
| Table 4 (raw-window control models) | `fit_toto_probes.py` (+ `toto_interp/fno.py`, `gbdt.py`) | `reviewer3_controls/reviewer3_summary.json` (parameter counts) |
| Table 5 (rotated within-BOOM support tests) | `run_structural_holdout_probes.py` | `reviewer3_controls/structural_holdout_all_seeds.csv` |
| Table 6 (Toto donor exchange) | `run_toto_paired_patch.py` | `reviewer_replications_5seed/paired_patch_per_resplit.csv` (+ `_manifest.csv`) |
| Table 7 (external transfer) | `run_toto_transfer.py` | `reviewer_replications_5seed/transfer_per_resplit.csv`, `transfer_per_dataset.csv` |
| Table 8 (MOMENT dynamic readouts) | `fit_toto_probes.py --label-group dynamic` | `reviewer_replications_5seed/moment_dynamic_per_resplit.csv`, `moment_random_dynamic_per_resplit.csv` |
| Table 9 (MOMENT matched interchange) | `run_moment_paired_interchange.py` | `reviewer_replications_5seed/moment_interchange_per_resplit.csv` (+ `_manifest.csv`) |
| Table 10 (full taxonomy-control matrix) | same as Table 1 | `reviewer3_controls/raw_control_all_seeds.csv`, `unconditional_selected_all_seeds.csv`, `layer_permuted_selected_all_seeds.csv` |
| Figure 1 (Toto vs. controls per label) | `scripts/make_camera_ready_fig_data.py` | same inputs as Table 1 |
| Figure 3 (per-resplit Toto vs. strongest raw-window model) | `scripts/make_camera_ready_fig_data.py` | same inputs as Table 1 |
| Figure 4 (probe check vs. forecast endpoint) | `scripts/plot_camera_ready_figures.py`, `scripts/make_camera_ready_fig_data.py` | `paired_patch_per_resplit.csv`, `moment_interchange_per_resplit.csv` |
| Cramér's V between labels | `compute_structural_confounding.py` | `reviewer3_controls/pairwise_cramers_v_all_seeds.csv` |

Every `reviewer3_controls/*` CSV and `reviewer3_summary.json` is written by
`scripts/summarize_reviewer3_suite.py` (`EXECUTION.md` is a hand-written
compute record) and every
`reviewer_replications_5seed/*` file by
`scripts/summarize_reviewer_replications.py` (Level B step 9).

`verify_paper_numbers.py` checks Tables 1–10, the plotted figure values,
the in-text means/half-widths/wins, the appendix series/window counts, and
a 48-cell supplementary win-count summary (424 checks).

## Compute notes (from results/reviewer3_controls/EXECUTION.md)

Taxonomy-control CPU stage (`run_reviewer3_cpu_followup.sh`, seeds 42–46): 16 CPU / 32 GB request; observed
1:07–1:25 elapsed, 18.7–21.3 GB peak RSS. Held-out + conditional tail:
2 CPU / 16 GB; 11:34–21:25, up to 8.8 GB. Layer-perm probes: 96 GB ask,
~79 GB observed. MOMENT random-init CPU tail: 100 GB ask, ~91 GB
observed. A10 GPU stages: MOMENT activations 21–37 min / ~29 GB; Toto
patch 3–4 min / 7–9 GB; MOMENT random activations 48–53 min / ~30 GB;
layer-perm activations 6–7 min / 42–49 GB; transfer ~1 min / ~2 GB;
MOMENT interchange 1–2 min / 2–3 GB.

## Data and model sources (downloaded at rerun; not included)

- BOOM benchmark: HuggingFace dataset `Datadog/BOOM` (2,807 series).
- Toto checkpoint: `Datadog/Toto-Open-Base-1.0` (Hub revisions unpinned).
- MOMENT-base checkpoint: `AutonLab/MOMENT-1-base` via `momentfm`.
- FEV: `autogluon/fev_datasets`. Transfer uses the 11 configurations
  listed in the paper; `stage_fev_safe_datasets.py` stages all
  configurations marked safe in `toto_interp/fev_tasks.py`. Rohlik
  configurations come from Rohlik competition data and must never be
  redistributed; only per-dataset aggregate transfer metrics (R2, RMSE,
  MAE, window counts) are bundled, with no series values.
- LSTF (ETTh1, ETTh2, weather, electricity): Google Drive bundles
  resolved by `toto_interp/lsf.py`.

No raw dataset files are included in this package; `results/` contains aggregate
experimental results and provenance records. Third-party
dataset and model terms apply to anything downloaded at rerun time.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and
upstream terms, and [docs/asset_provenance.md](docs/asset_provenance.md)
for recorded asset revisions.

## Known limitations (factual)

- Per-view (layer × token × pooling) probe metrics exist only on cluster
  scratch; the package carries validation-selected rows only.
- Toto taxonomy-suite window counts are not recorded in the released
  artifacts (MOMENT-suite counts are in `moment_manifest.csv`).
- Hub revisions (BOOM snapshot, Toto/MOMENT checkpoints) are unpinned;
  reruns download current revisions.
- `momentfm` has no recorded version pin (see `requirements.txt`).
