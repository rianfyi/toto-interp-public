# Asset provenance and reproduction limits

## Released results

`results/E2E_AUDIT.json` records the audit of the packaged aggregates.
`reproduce/verify_paper_numbers.py` checks 424 reported values and summaries
against those aggregates. A successful check validates the released numbers;
it does not constitute an independent rerun of the original GPU experiments.

## Original experiment revisions

The released manifests do not record exact Hugging Face revisions for the
original BOOM snapshot, Toto checkpoint, MOMENT checkpoint, or FEV data.
Historical checksums for the downloaded LSTF archives are also unavailable.
SHA-256 values in the released audit and manifests describe result files,
probe artifacts, or series splits; they are not upstream repository revisions.

The requirements file records the available package-version pins and marks
unknown environment details. The release does not substitute today's asset
revisions for missing historical records. The loaders still use their existing
download behavior, so a fresh rerun can obtain different upstream revisions.

## Revisions recorded during the September 27, 2026 rerun

The authors' rerun report records the following identifiers:

| Asset | Recorded revision | Scope of the record |
| --- | --- | --- |
| `Datadog/Toto-Open-Base-1.0` | `0411ceb27bdf7fc3e4892e99edc8ad08192dc3c5` | Checkpoint identified in the September rerun report. |
| `AutonLab/MOMENT-1-base` | `5e44b0ea26376a176360f87831124e018f876d96` | Checkpoint identified in the September rerun report. |
| `Datadog/BOOM` | `69325b544c45ff0d6c43c7a99c49a6601a01725b` | Then-current upstream revision observed during the rerun; not proof of the exact cached dataset snapshot consumed. |

These identifiers were transcribed from the authors' rerun report, not
recovered from the original run manifests. The report is inconsistent about
whether the model revisions match the original July runs. They therefore must
not be treated as confirmed original-run pins. No corresponding FEV revision
or LSTF archive checksum was recovered.

As described in the README, the full GPU rerun matched 346 of the 424 checks
at printed precision. This release preserves the original audited aggregates
and discloses the rerun differences. It makes no claim of bitwise GPU
reproducibility.
