# Asset provenance

## Released results

`results/` holds the audited five-resplit aggregates (seeds 42–46) from which
the paper's tables and figures are computed. `results/E2E_AUDIT.json` records,
for each of the 13 aggregate files, the seeds (each exactly once per cell), the
row and cell counts, the file's SHA-256, and a `complete` status; the recorded
hashes match the released files. `reproduce/verify_paper_numbers.py`
recomputes the 424 checked values described in the
[README](../README.md#check-the-papers-numbers-cpu) from these
files and compares each with `reproduce/expected_numbers.csv`
(`PASS 424  FAIL 0`). The README's
[rerun section](../README.md#rerun-the-pipeline-gpu-cluster) lists the compute
each pipeline stage used.

## Upstream assets

No dataset files or model weights are bundled. The code below downloads each
asset from its source; the README lists the staging commands, and
`requirements.txt` pins the package versions.

| Asset | Source | Loaded by | Files read |
| --- | --- | --- | --- |
| BOOM | Hugging Face dataset [`Datadog/BOOM`](https://huggingface.co/datasets/Datadog/BOOM) | `toto_interp/boom.py`; staged by `scripts/stage_assets.py` | `dataset_taxonomy.json` and the folders of the 2,700 series selected by the seed 42–46 splits |
| Toto-Open-Base-1.0 | Hugging Face model [`Datadog/Toto-Open-Base-1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | `toto_interp/loader.py` (`load_toto_with_fallback`, through `toto-ts`) | `config.json`, `model.safetensors` |
| MOMENT-1-base | Hugging Face model [`AutonLab/MOMENT-1-base`](https://huggingface.co/AutonLab/MOMENT-1-base) | `toto_interp/moment_loader.py` (`load_moment_with_fallback`, through `momentfm`) | `config.json`, `model.safetensors` |
| FEV | Hugging Face dataset [`autogluon/fev_datasets`](https://huggingface.co/datasets/autogluon/fev_datasets), by configuration name | `toto_interp/transfer.py` (`load_fev_dataset`); staged by `scripts/stage_fev_safe_datasets.py` | the configurations flagged `safe_for_paper` in `toto_interp/fev_tasks.py` |
| LSTF (ETTh1, ETTh2, electricity, weather) | Time-Series-Library Google Drive bundles listed in [lsf_setup.md](lsf_setup.md) | `toto_interp/lsf.py`; `scripts/download_lsf_datasets.py` | the ETT, electricity, and weather CSV files |

The external-transfer results cover the 11 FEV configurations listed in the
paper appendix and in
`results/moment_exchange_transfer/transfer_per_dataset.csv`. The LSTF
downloader fetches the bundles from their fixed links with `gdown` and checks
the expected CSV layout.

## Hugging Face revisions

The loaders read each repository's default branch (`main`). The September 2026
GPU rerun resolved Toto-Open-Base-1.0 at
`0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5`, MOMENT-1-base at
`5e44b0ea26376a176360f87831124e018f876d96`, and BOOM at
`69325b544c45ff0d6c43c7a99c49a6601a01725b`. The public Hub commit histories
(checked in September 2026) give the last-change dates of the files the
pipeline reads:

| Repository | Files read | Last changed on the Hub | `main` head (commit date) |
| --- | --- | --- | --- |
| `Datadog/Toto-Open-Base-1.0` | `config.json`; `model.safetensors` | 2025-05-15; 2025-05-06 | `0411ceb` (2026-05-14) |
| `AutonLab/MOMENT-1-base` | `config.json`, `model.safetensors` | 2024-10-12 | `5e44b0e` (2025-03-26) |
| `Datadog/BOOM` | `dataset_taxonomy.json`; series folders | 2025-05-14; 2025-05-19 | `69325b5` (2025-09-08) |
| `autogluon/fev_datasets` | `README.md` (configuration index); folders of the flagged configurations | 2026-01-28; 2025-09-05 to 2025-09-30 | `f71c0ff` (2026-01-28) |

All of these dates precede the July 2026 runs that produced the released
results, so the files these loaders read from `main` did not change between
those runs and the September 2026 rerun. The rerun also confirmed that the
BOOM `dataset_taxonomy.json` staged for the July runs is byte-identical to the
Hub copy.

## GPU rerun

In September 2026 the authors reran the full five-resplit pipeline with this
code and the package versions pinned in `requirements.txt`. Every stage
completed, and the regenerated `E2E_AUDIT.json` reports the same seeds, row
counts, and `complete` status for every aggregate. CPU-only computations, such
as the Cramér's V values, match exactly. Values from GPU-trained models or
GPU activations differ slightly across runs, as is usual for GPU
floating-point computation; the README's
[rerun section](../README.md#what-to-expect-from-a-rerun) gives the
check-level comparison.
