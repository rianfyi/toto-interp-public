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

The upstream license and notice were checked at
[`9813a2e6d57f9d73bbf50cbe02d7d1e9d7029717`](https://github.com/DataDog/toto/tree/9813a2e6d57f9d73bbf50cbe02d7d1e9d7029717).
This is the attribution-review revision, not a claim about the exact upstream
revision used when the utilities were first adapted or the experiments ran.

## Dependencies

The package imports Toto (`toto-ts`), MOMENT (`momentfm`), and the packages
listed in [requirements.txt](requirements.txt) and
[pyproject.toml](pyproject.toml). Their implementations and distributions are
not bundled here. Each dependency retains its own license and notices, which
are supplied by its original distributor. The MIT license for this repository
does not relicense those dependencies.

## Datasets and model weights

No raw time series, dataset copies, activations, or model weights are included
in this release. The files in `results/` are aggregate experimental outputs
and provenance records. Downloaded assets retain their upstream terms:

- [BOOM](https://huggingface.co/datasets/Datadog/BOOM) and
  [Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0)
  are obtained from Datadog.
- [MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base) is obtained
  from the MOMENT authors.
- [FEV datasets](https://huggingface.co/datasets/autogluon/fev_datasets)
  include several original data sources, whose individual terms apply.
- LSTF data are obtained using the sources in [docs/lsf_setup.md](docs/lsf_setup.md).

The source-specific terms for some FEV configurations, LSTF electricity and
weather data, and Rohlik competition data remain unresolved in the released
records. In particular, the package does not grant rights to redistribute
Rohlik data. The `safe_for_paper` selector describes the experimental task
selection, not legal clearance or redistribution permission.

This notice does not resolve those uncertainties or change checklist item 12.
Users must establish the applicable access and use terms before downloading
or using an upstream asset. See [asset provenance](docs/asset_provenance.md)
for the limits of the recorded revisions.
