# LSTF dataset setup

The long-sequence time-series forecasting (LSTF) CSV datasets, called LSF in
the code and command-line flags, are not bundled in this repo and are not
shipped inside `toto-ts`. They come from the public long-sequence forecasting
benchmark setup used by Toto.

The original public source referenced by Toto is the
[Time-Series-Library](https://github.com/thuml/Time-Series-Library), with the
preprocessed CSV bundles hosted on Google Drive.

## Download sources

| Dataset bundle | Public link |
| --- | --- |
| ETT (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`) | [Google Drive](https://drive.google.com/file/d/1bnrv7gpn27yO54WJI-vuXP5NclE5BlBx/view?usp=drive_link) |
| Electricity | [Google Drive](https://drive.google.com/file/d/1FHH0S3d6IK_UOpg6taBRavx4MragRLo1/view?usp=drive_link) |
| Weather | [Google Drive](https://drive.google.com/file/d/1nXdMIJ7K201Bx3IBGNiaNFQ6FzeDEzIr/view?usp=drive_link) |

## Expected directory layout

After download and extraction, Toto expects this layout:

```text
data/
└── lsf_datasets/
    ├── ETT-small/
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── electricity/
    │   └── electricity.csv
    └── weather/
        └── weather.csv
```

## Recommended setup

This repo includes a helper that downloads the archives, extracts them, and
normalizes them into the exact structure above:

```bash
python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets
```

To only validate an already-downloaded local layout:

```bash
python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets --validate-only
```

To download only a subset:

```bash
python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets --datasets ett weather
```

## Using LSTF in transfer runs

With the files in place, run:

```bash
python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer --dataset lsf --lsf-path data/lsf_datasets --lsf-datasets ETTh1 electricity
```

To fetch missing files automatically, add `--download-lsf`:

```bash
python scripts/run_toto_transfer.py --probe-dir runs/probes --output-dir runs/transfer --dataset lsf --lsf-path data/lsf_datasets --lsf-datasets ETTh1 electricity --download-lsf
```
