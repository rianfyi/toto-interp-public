# Third-party notices

The authors' original code, documentation, and aggregate experimental results
in this repository are provided under the [MIT License](LICENSE). This grant
covers only rights held by the toto-interp contributors. It does not replace
licenses for third-party material, datasets, model weights, or dependencies.

## Material adapted from Datadog Toto

The following files contain material adapted from
[Datadog Toto](https://github.com/DataDog/toto), distributed under Apache-2.0:

| Local file | Upstream source and modifications |
| --- | --- |
| `toto_interp/fev_tasks.py` | `toto/evaluation/fev/tasks.yaml` and `toto/evaluation/fev/evaluate.py`; converted into a Python task registry with audit-specific selectors. |
| `toto_interp/boom.py` | The upstream notebook helper for downloading BOOM; ported into this standalone package and integrated with its audit utilities. |

The upstream-derived material remains subject to Apache-2.0. The original
[Apache-2.0 license text](licenses/Apache-2.0.txt) and
[Datadog Toto NOTICE](licenses/Datadog-Toto-NOTICE) accompany this release.
The adapted files carry notices describing the changes.

Upstream notice:

```text
Datadog Toto
Copyright [2005-Present] Datadog, Inc.
This product includes software developed at Datadog (https://www.datadoghq.com/).
```

The bundled license text and notice match those in DataDog/toto at
[`9813a2e6d57f9d73bbf50cbe02d7d1e9d7029717`](https://github.com/DataDog/toto/tree/9813a2e6d57f9d73bbf50cbe02d7d1e9d7029717).

## Dependencies

The package imports Toto (`toto-ts`), MOMENT (`momentfm`), and the packages
listed in [requirements.txt](requirements.txt) and
[pyproject.toml](pyproject.toml). Their implementations and distributions are
not bundled here. Each dependency retains its own license and notices, which
are supplied by its original distributor. The MIT license for this repository
does not relicense those dependencies.

## Datasets and model weights

This repository contains no dataset files, raw time series, activations, or
model weights. The files in `results/` are aggregate experimental outputs and
provenance records. The scripts download each upstream asset from its source,
and each asset is obtained and used under that source's terms:

| Asset | Source | Terms |
| --- | --- | --- |
| Toto-Open-Base-1.0 | [Hugging Face model card](https://huggingface.co/Datadog/Toto-Open-Base-1.0) (Datadog) | Apache-2.0 |
| BOOM | [Hugging Face dataset card](https://huggingface.co/datasets/Datadog/BOOM) (Datadog) | Apache-2.0 |
| MOMENT-1-base | [Hugging Face model card](https://huggingface.co/AutonLab/MOMENT-1-base) (AutonLab) | MIT |
| FEV configurations | [Hugging Face dataset card](https://huggingface.co/datasets/autogluon/fev_datasets) (`autogluon/fev_datasets`) | The card lists each configuration's original source, directs users to those sources for licensing and citation terms, and provides the datasets only for research purposes unless otherwise specified. |
| ETTh1, ETTh2 | [ETDataset](https://github.com/zhouhaoyi/ETDataset), through the Time-Series-Library bundle | CC BY-ND 4.0 |
| LSTF electricity and weather | Time-Series-Library bundles ([docs/lsf_setup.md](docs/lsf_setup.md)) | Terms of the original data providers |

The Rohlik configurations (`rohlik_sales_1D`, `rohlik_orders_1D`) come from
Rohlik's Kaggle forecasting competitions
([orders](https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge),
[sales](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2)),
whose rules set the terms for that data. This repository contains none of that
data and grants no rights to it; the released results include only aggregate
transfer metrics for `rohlik_sales_1D`.

The `safe_for_paper` flag in `toto_interp/fev_tasks.py` carries over the
per-dataset flags of the upstream Toto FEV evaluation (`DATASETS` in
`toto/evaluation/fev/evaluate.py`), where `True` marks the datasets described
as "not contaminated by LOTSA"; the transfer runs select their FEV tasks with
it.

See [docs/asset_provenance.md](docs/asset_provenance.md) for where each asset
is loaded and its Hugging Face revisions.
