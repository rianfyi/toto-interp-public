# Reproducing the results

This page has the details for rerunning the pipeline that produced `results/`. To check the paper's numbers from the released results on a CPU, see the [quick start](../README.md#quick-start-check-the-papers-numbers) in the README.

- [Setup](#setup)
- [Run the pipeline](#run-the-pipeline)
- [Stage 9: summarize, audit and verify](#stage-9-summarize-audit-and-verify)
- [What the checks and figure scripts cover](#what-the-checks-and-figure-scripts-cover)
- [What to expect from a rerun](#what-to-expect-from-a-rerun)
- [Reproducibility notes](#reproducibility-notes)
- [Running on a Slurm cluster](#running-on-a-slurm-cluster)
- [Where each paper result comes from](#where-each-paper-result-comes-from)

## Setup

### Python environment

Use Python 3.12. Install the `torch` 2.7.0 build that matches your CUDA version first (our rerun used `torch 2.7.0+cu126`), then:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements.txt
pip install -e .
pip install --no-deps momentfm==0.1.4   # its own dependency pins conflict with requirements.txt
```

`scripts/run_pipeline.sh` uses `.venv-gpu/bin/python` when it exists. The unit tests run in the same environment: `python -m pytest -q tests`.

### Data and models

Run these once, from the repository root, on a machine with network access. Log in to Hugging Face first (`huggingface-cli login`, or set `HF_TOKEN`): staging fetches about 2,700 BOOM series folders, and unauthenticated downloads hit rate limits.

```bash
export HF_HOME="$PWD/.cache/huggingface"
python scripts/stage_assets.py --snapshot-path data/boom_snapshot --seeds 42 43 44 45 46 --max-series-per-split 500
python scripts/stage_fev_safe_datasets.py
python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets
python -c "from toto_interp.moment_loader import load_moment_with_fallback; load_moment_with_fallback()"  # caches MOMENT-1-base
```

This puts the BOOM series in `data/boom_snapshot`, the LSTF files in `data/lsf_datasets`, and the model weights and FEV datasets in the Hugging Face cache under `.cache/huggingface`, which is where the pipeline looks for them. The pipeline itself runs offline.

| Asset | Source | Used for |
|---|---|---|
| BOOM (2,807 series) | Hugging Face [`Datadog/BOOM`](https://huggingface.co/datasets/Datadog/BOOM) | All probes, baselines and exchanges |
| Toto-Open-Base-1.0 | Hugging Face [`Datadog/Toto-Open-Base-1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | The model under study |
| MOMENT-1-base | Hugging Face [`AutonLab/MOMENT-1-base`](https://huggingface.co/AutonLab/MOMENT-1-base), via `momentfm` | Cross-model comparison |
| FEV (11 configurations) | Hugging Face [`autogluon/fev_datasets`](https://huggingface.co/datasets/autogluon/fev_datasets) | Zero-shot transfer |
| LSTF (ETTh1, ETTh2, weather, electricity) | Time-Series-Library bundles on Google Drive ([setup](lsf_setup.md)) | Zero-shot transfer |

Each dataset and checkpoint is used under its own terms. The Rohlik FEV configurations come from Rohlik's Kaggle competition, whose rules set their terms; this repository stores only aggregate transfer metrics for them. [asset_provenance.md](asset_provenance.md) records the exact Hugging Face revisions our rerun resolved.

## Run the pipeline

```bash
bash scripts/run_pipeline.sh                              # all stages (1 to 9), seeds 42 to 46
bash scripts/run_pipeline.sh --stages "1 2" --seeds "42"  # only some stages or seeds
bash scripts/run_pipeline.sh --dry-run                    # print every command without running it
```

These environment variables change the defaults:

| Variable | Default | What it sets |
|---|---|---|
| `PYTHON` | `.venv-gpu/bin/python` if it exists, otherwise `python3` | The Python interpreter |
| `RUNS` | `runs` | The folder that holds every stage's outputs |
| `HF_HOME` | `<repo>/.cache/huggingface` | The Hugging Face cache with the staged models and datasets |

For example, `RUNS=/scratch/toto-runs bash scripts/run_pipeline.sh` keeps all outputs on a scratch disk.

Later stages read what earlier ones wrote: stages 2, 3, 6 and 8 use the Toto activations from stage 1, and stage 7 uses the MOMENT activations and probes from stage 5. Run the stages in order, or use `--stages` to resume once the earlier outputs exist. Outputs go to these folders under `runs/`:

| Folder | Written by |
|---|---|
| `taxonomy_suite/` | stages 1 to 3 |
| `layer_permuted_pretrained/` | stage 4 |
| `moment_suite/` | stages 5 and 7 |
| `toto_exchange_transfer/` | stages 6 and 8 |
| `summary/` | stage 9 |

### Stages

Time and peak memory are per seed, as recorded in the reported runs on a cluster, where each GPU step had one NVIDIA A10 (24 GB) and each CPU job in stages 2 to 8 had 16 cores (stage 3 had 2). Peak memory is host RAM. Where memory was not recorded, the memory our launcher requested is shown as a guide.

| Stage | Step | Runs on | Paper items | Time | Peak memory |
|---|---|---|---|---|---|
| 1 | Toto activations (pretrained and random-init), Cramér's V, raw-window models | GPU | Tables 1, 4, 10; Figs. 1, 3; §5.2 | not recorded | not recorded (requested 32 GB) |
| 2 | Linear probes on Toto activations | CPU | Tables 1, 10; Figs. 1, 3 | 1:07 to 1:25 h | 18.7 to 21.3 GB |
| 3 | Held-out-combination tests and common-support probes | CPU | Tables 2, 5 | 11 to 21 min | up to 8.8 GB |
| 4 | Block-permuted Toto activations | GPU | Tables 1, 10 | about 7 min | 42 to 49 GB |
| 4 | Probes on the block-permuted activations | CPU | Tables 1, 10 | 39 min to 1:19 h | about 79 GB |
| 5 | Pretrained MOMENT activations | GPU | Tables 3, 8 | 21 to 37 min | about 29 GB |
| 5 | Random-init MOMENT activations | GPU | Tables 3, 8 | 48 to 53 min | 29 to 31 GB |
| 5 | Taxonomy and dynamic probes on pretrained MOMENT | CPU | Tables 3, 8 | not recorded | not recorded (requested 100 GB) |
| 5 | Taxonomy and dynamic probes on random-init MOMENT, then provenance | CPU | Tables 3, 8 | not recorded | about 91 GB |
| 6 | Toto donor exchange | GPU | Table 6; Fig. 4 (left) | 3 to 4 min | 6.7 to 9.0 GB |
| 7 | MOMENT matched interchange | GPU | Table 9; Fig. 4 (right) | 1 to 2 min | 1.7 to 3.0 GB |
| 8 | Toto dynamic probes | CPU | Table 7 | not recorded | not recorded (requested 120 GB) |
| 8 | Zero-shot transfer | GPU | Table 7 | about 1 min | 1.4 to 1.8 GB |
| 9 | Summarize, audit and verify | CPU | all | not recorded | not recorded |

For five seeds run back to back, the steps with recorded times add up to about 16.5 to 24 hours, and the steps marked "not recorded" come on top of that. Times on your machine will depend on its GPU and core count.

<details>
<summary><b>Main settings of each stage</b></summary>

The runner runs the same commands, with the same settings, as the Slurm launchers in `examples/slurm/` (named at the end of each item), one seed at a time. `SEED` is the seed being run.

1. `run_taxonomy_suite.py --seed SEED --raw-only` runs `dump_toto_activations.py --context-length 1024 --max-series-per-split 500 --max-windows-per-series 4` for pretrained and random-init Toto, then `compute_structural_confounding.py`, then the FNO, CNN, Transformer and GBDT raw-window models (`fit_toto_probes.py`; width 32, 3 layers, 16 modes, 20 epochs, batch 16). Slurm launcher: `run_taxonomy_suite.sh`.
2. `run_taxonomy_probes.py --seed SEED --skip-post-controls --reuse-existing` fits the taxonomy linear probes on pretrained and random-init activations. Slurm launcher: `run_taxonomy_probes.sh`.
3. `run_holdout_grid.py --source {pretrained,random_init} --seed SEED --rotation SEED-42` runs the held-out-combination tests (`run_structural_holdout_probes.py`) and the common-support probes (`run_conditional_probes.py`). Slurm launcher: `run_holdout_grid.sh`.
4. `dump_toto_activations.py --layer-permutation-seed SEED --weight-source layer_permuted_pretrained`, with both pooling modes, then taxonomy probe fits. Slurm launchers: `run_layer_permuted_pretrained_a10.sh`, then `run_layer_permuted_pretrained_probe_followup.sh`.
5. `dump_moment_activations.py --seed SEED --seq-len 512 --max-series-per-split 500 --max-windows-per-series 4 --layers 3 6 9 11 --token-positions all_context --pooling-modes series_mean` for pretrained and random-init MOMENT, then taxonomy and dynamic probe fits. `enrich_moment_run_provenance.py` then adds split and evaluation provenance to the random-init run. Pretrained MOMENT gets the same step at the start of stage 7, as in `run_moment_interchange.sh`. Slurm launchers: `run_moment.sh`, `run_moment_random.sh`, `run_moment_dynamic.sh`, `run_moment_random_probe_followup.sh`.
6. Fits the layer-11, all-context, series-mean future-burstiness probe, then runs `run_toto_paired_patch.py --split test --split-seed SEED --seeds SEED --context-length 1024 --max-series 500 --max-windows-per-series 4 --max-windows-eval 2000 --num-pairs 40 --num-samples 16 --blend {0.25,0.5,1.0}`. Slurm launcher: `run_toto_donor_exchange.sh`.
7. `run_moment_paired_interchange.py --split-seed SEED --sampling-seed SEED --num-pairs 40 --blends 0.25 0.5 1.0 --high-quantile 0.75 --low-quantile 0.25 --null-match-k 5 --secondary-endpoint auto`. Slurm launcher: `run_moment_interchange.sh`.
8. Fits the Toto dynamic probes that transfer applies, then runs `run_toto_transfer.py --dataset both --fev-safe-only --max-series 100 --max-windows-per-series 4 --lsf-datasets ETTh1 ETTh2 weather electricity --context-length 1024 --validation-select-one-per-label --require-eval-provenance`. Slurm launchers: `run_toto_dynamic.sh`, then `run_transfer.sh`.
9. The commands in the [next section](#stage-9-summarize-audit-and-verify) (no launcher).

</details>

## Stage 9: summarize, audit and verify

Stage 9 runs these commands (with `runs` replaced by `$RUNS` if you set it). You can also run them yourself, for example after running the other stages on a cluster.

```bash
python scripts/summarize_taxonomy_suite.py \
  --runs-root runs/taxonomy_suite \
  --layer-permuted-runs-root runs/layer_permuted_pretrained \
  --output-dir runs/summary/toto_taxonomy
python scripts/summarize_moment_exchange_transfer.py \
  --runs-root runs/toto_exchange_transfer \
  --moment-runs-root runs/moment_suite \
  --output-dir runs/summary/moment_exchange_transfer \
  --seeds 42 43 44 45 46
python scripts/audit_results.py \
  --moment-exchange-transfer-summary-dir runs/summary/moment_exchange_transfer \
  --taxonomy-summary-dir runs/summary/toto_taxonomy \
  --output runs/summary/E2E_AUDIT.json
bash reproduce/verify_against.sh runs/summary
```

- The audit fails unless every expected cell is present for all five seeds, so it only passes after a full run. Its output should report `complete` for every file, as in `results/E2E_AUDIT.json`.
- `verify_against.sh` runs the same 424 checks as the CPU verifier on your rerun's results. It exits with an error if any check fails, which is expected after a GPU rerun (see [below](#what-to-expect-from-a-rerun)).

## What the checks and figure scripts cover

The 424 checks cover Tables 1 to 10, the values plotted in Figures 1, 3 and 4, the means, half-widths and win counts quoted in the text, and the appendix series and window counts. They also include a 48-cell win-count summary and 11 derived margins.

The two figure scripts from the README quick start (`plot_camera_ready_figures.py`, then `make_camera_ready_fig_data.py`) read `results/` and write these files to `paper/neurips2026/figures/`:

- `data/*.csv`: the plotted values.
- `tikz/gen/`: the TikZ coordinates for Figures 1, 3 and 4.
- `cr_figure_values.csv`: every plotted value with the file and columns it came from.
- Matplotlib previews of the same values.

## What to expect from a rerun

Values from GPU-trained models or GPU activations differ slightly between runs. In our full rerun of this code (September 2026):

- Cramér's V and the other CPU-only values matched exactly.
- 346 of the 424 verifier checks passed. The other values moved by a median of about 0.002, and at most about 0.024.
- Three near-tie selections changed: the domain win count against the strongest raw-window model (2/5 became 3/5), the strongest raw-window model for cardinality (FNO and CNN swapped), and one random-init common-support view (seed 46).

The figure scripts check the paper's printed values, so they stop on rerun outputs that differ from them.

## Reproducibility notes

- **Seeds.** Each seed from 42 to 46 defines one series-disjoint train/validation/test resplit of BOOM. Activation views are chosen on validation series and evaluated once on test series.
- **GBDT backend.** `fit_toto_probes.py` defaults `--gbdt-backend` to `hist_gradient_boosting`. In the reported runs the backend was set to `auto`, which used scikit-learn's HistGradientBoosting because XGBoost wasn't installed. The new default makes that choice explicit. It is the only functional change from the code that produced the results.
- **Validation-selected rows.** `results/` holds the validation-selected probe rows behind every reported number.
- **Window counts.** `moment_manifest.csv` and `paired_patch_manifest.csv` in `results/moment_exchange_transfer/` record the window counts per split for the MOMENT suite and the Toto donor exchange.
- **Model and data revisions.** The loaders read each Hugging Face repository's default branch. [asset_provenance.md](asset_provenance.md) records the revisions our rerun resolved and when the files the pipeline reads last changed.

## Running on a Slurm cluster

[`examples/slurm/`](../examples/slurm/) has the launchers that produced the reported results. Each pipeline launcher is a Slurm array over seeds 42 to 46 and runs one step of the [stage table](#stages). To use them:

1. Export `REPO` (the absolute path of this checkout) and `SOFTWARE_STACK` (the environment module that provides Python), and run `mkdir -p logs`. The launchers use the `.venv-gpu` environment from [setup](#python-environment).
2. Submit each launcher with your site's account and partition, for example `sbatch --account=<account> --partition=<partition> examples/slurm/run_taxonomy_suite.sh`.
3. Submit them in stage order, or chain them with Slurm dependencies (the launcher headers show how). GPU launchers request one NVIDIA A10 (`--gres=gpu:A10:1`); override this if your cluster names its GPUs differently. `RUNS_ROOT` changes where a launcher writes.
4. Run stage 9 with `bash scripts/run_pipeline.sh --stages 9`, or with the commands above.

`setup_cluster.sh`, run on a login node, creates `.venv-gpu`, installs this package with `pip install -e`, and stages BOOM and Toto. It does not install the `requirements.txt` pins or `momentfm`, and it does not stage FEV, LSTF or MOMENT, so run those steps from [setup](#setup) as well. `enrich_moment_random_provenance.sh` reruns only the random-init MOMENT provenance step. Each launcher's header lists the `${...}` placeholders for site-specific values.

## Where each paper result comes from

Each row gives the script that computes a paper item and the files in `results/` that hold its numbers.

| Paper item | Script(s) | Results file(s) |
|---|---|---|
| Table 1: Toto vs. input and backbone baselines | `run_taxonomy_suite.py`, `run_taxonomy_probes.py`, `fit_toto_probes.py`; block permutation: stage 4 | `toto_taxonomy/unconditional_selected_all_seeds.csv`, `toto_taxonomy/raw_control_all_seeds.csv`, `toto_taxonomy/layer_permuted_selected_all_seeds.csv` |
| Table 2: common-support probes | `run_conditional_probes.py` | `toto_taxonomy/conditional_all_seeds.csv` |
| Table 3: MOMENT-base taxonomy readouts | `dump_moment_activations.py`, `fit_toto_probes.py` | `moment_exchange_transfer/moment_per_resplit.csv`, `moment_exchange_transfer/moment_random_structural_per_resplit.csv` |
| Table 4: raw-window baseline models | `fit_toto_probes.py`, `toto_interp/fno.py`, `toto_interp/gbdt.py` | `toto_taxonomy/taxonomy_summary.json` (parameter counts) |
| Table 5: held-out-combination tests | `run_structural_holdout_probes.py` | `toto_taxonomy/structural_holdout_all_seeds.csv` |
| Table 6: Toto donor exchange | `run_toto_paired_patch.py` | `moment_exchange_transfer/paired_patch_per_resplit.csv`, `moment_exchange_transfer/paired_patch_manifest.csv` |
| Table 7: zero-shot coordination-probe transfer | `run_toto_transfer.py` | `moment_exchange_transfer/transfer_per_resplit.csv`, `moment_exchange_transfer/transfer_per_dataset.csv` |
| Table 8: MOMENT dynamic readouts | `fit_toto_probes.py --label-group dynamic` | `moment_exchange_transfer/moment_dynamic_per_resplit.csv`, `moment_exchange_transfer/moment_random_dynamic_per_resplit.csv` |
| Table 9: MOMENT matched interchange | `run_moment_paired_interchange.py` | `moment_exchange_transfer/moment_interchange_per_resplit.csv`, `moment_exchange_transfer/moment_interchange_manifest.csv` |
| Table 10: all raw-window and backbone baselines | same as Table 1 | same as Table 1 |
| Figures 1 and 3 | `make_camera_ready_fig_data.py` | same as Table 1 |
| Figure 4 | `plot_camera_ready_figures.py`, `make_camera_ready_fig_data.py` | `moment_exchange_transfer/paired_patch_per_resplit.csv`, `moment_exchange_transfer/moment_interchange_per_resplit.csv` |
| Cramér's V between labels (§5.2) | `compute_structural_confounding.py` | `toto_taxonomy/pairwise_cramers_v_all_seeds.csv` |

`summarize_taxonomy_suite.py` writes every file in `results/toto_taxonomy/`, and `summarize_moment_exchange_transfer.py` writes every file in `results/moment_exchange_transfer/`.

Some file names use the code's terms rather than the paper's:

| In the code | In the paper |
|---|---|
| `raw_control` | raw-window models |
| `layer_permuted` | block permutation |
| `conditional` | common-support probes |
| `structural_holdout` | held-out-combination tests |
| `paired_patch` | donor exchange |
| `frequency_bucket` | cadence |
