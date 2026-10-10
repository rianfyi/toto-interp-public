# What Does an Observability Foundation Model Know?

Code and results for our NeurIPS 2026 paper (poster). We ask what a linear probe can read out of Toto, a forecasting foundation model for observability metrics, about the series it sees, whether that goes beyond what the raw input already shows, and whether its forecasts actually use it.

<p align="center">
  <a href="https://arxiv.org/abs/2610.05577">Paper (arXiv:2610.05577)</a>
</p>

![Method overview: linear probes on frozen Toto residuals compared with raw-window and backbone baselines (a), and the donor exchange that tests whether moving a readout moves the forecast (b)](assets/fig2_method.png)

## Quick start: check the paper's numbers

This needs no GPU and no data downloads. `results/` holds the result tables behind every number in the paper, and one script recomputes the paper's numbers from them in a few seconds.

```bash
git clone --branch neurips-2026-v1 https://github.com/DhyeyMavani2003/toto-interp.git
cd toto-interp
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib
python reproduce/verify_paper_numbers.py
```

It checks 424 values (every table, the plotted figure values and the statistics quoted in the text) against `reproduce/expected_numbers.csv`, and should end with:

```
PASS 424  FAIL 0  TOTAL 424
ALL PAPER NUMBERS VERIFIED
```

To rebuild the data behind Figures 1, 3 and 4, run these two scripts in this order. They write to `paper/neurips2026/figures/` and stop if any value differs from the paper.

```bash
python scripts/plot_camera_ready_figures.py
python scripts/make_camera_ready_fig_data.py
```

## Rerun the full pipeline

The pipeline recomputes everything in `results/` from scratch for all five data splits (seeds 42 to 46). You need one machine with a CUDA GPU (we used an NVIDIA A10) and about 128 GB of RAM: the largest recorded peak was about 91 GB, and one 16-core probe-fitting step was given 120 GB on our cluster. Plan for a day or more.

1. **Set up once.** Build the Python 3.12 environment in `.venv-gpu`, then download the data and model weights while you have network access. The commands are in [docs/reproducing.md](docs/reproducing.md#setup).
2. **Run it** from the repository root:

   ```bash
   bash scripts/run_pipeline.sh                              # all stages (1 to 9), seeds 42 to 46
   bash scripts/run_pipeline.sh --stages "1 2" --seeds "42"  # only some stages or seeds
   bash scripts/run_pipeline.sh --dry-run                    # print every command without running it
   ```

The pipeline uses the data and model files you downloaded in step 1. It writes everything under `runs/`, with the final tables in `runs/summary/`, and leaves the released `results/` untouched. The last stage summarizes these outputs, audits them, and runs the same 424 checks on the new results. Values that come from GPU computations shift slightly from run to run, so expect most checks to pass but not all: 346 of 424 passed in our own full rerun.

[docs/reproducing.md](docs/reproducing.md) explains each stage, how long it took, its settings and what to expect from a rerun. If you work on a Slurm cluster, the launchers we originally used are in [`examples/slurm/`](examples/slurm/).

## Results

Across five series-disjoint resplits of the BOOM benchmark, a linear probe on Toto's frozen residual stream recovers cadence (short vs. medium) and metric type better than every raw-window model we tested. Domain is nearly tied, and cardinality is recovered far better from the raw window.

| Label | Toto | Strongest raw-window model | Paired gap | Resplits Toto wins |
|---|---|---|---|---|
| Cadence | 0.766 ± 0.024 | FNO 0.633 ± 0.047 | +0.133 ± 0.063 | 5/5 |
| Metric type | 0.545 ± 0.037 | GBDT 0.498 ± 0.007 | +0.047 ± 0.036 | 5/5 |
| Domain | 0.484 ± 0.011 | GBDT 0.482 ± 0.040 | +0.002 ± 0.040 | 2/5 |
| Cardinality | 0.471 ± 0.029 | FNO 0.961 ± 0.016 | -0.490 ± 0.044 | 0/5 |

<sub>Held-out test macro-F1, mean ± Student-t(4) 95% half-width over five resplits (paper Table 1).</sub>

![Held-out test macro-F1 of Toto, its baselines, and the shuffled-label floor for each taxonomy label](assets/fig1_label_profile.png)

- **Trained weights matter.** Cadence and metric type also beat Toto with random weights and Toto with its blocks shuffled, in all five resplits.
- **MOMENT-base** shows related cadence, metric-type and domain readouts on the same resplits.
- **Readout is not use.** Swapping in residuals from high-burst donor series moves a future-burstiness readout as intended, but the forecasts do not become consistently burstier than with a randomized donor.
- **Zero-shot transfer is negative.** A coordination probe trained on BOOM has negative R² on the external benchmarks we tested.

**Scope.** These are claims about linear recoverability from one representation with one probe family, on BOOM. Five resplits of one corpus measure split-to-split variability, not uncertainty over independent data, and Toto's pretraining overlap with BOOM is unknown. The full protocol is in §3, §4 and App. A of the paper.

## Repository layout

<details>
<summary>Show the folder tree</summary>

```
.
├── toto_interp/                    # analysis library: probes, labels, baselines, interventions, transfer
├── scripts/                        # run_pipeline.sh, pipeline stages, summaries, audit, figure data, data staging
├── results/
│   ├── toto_taxonomy/              # Toto probes and baselines (Tables 1, 2, 4, 5, 10)
│   ├── moment_exchange_transfer/   # MOMENT, donor exchange, transfer (Tables 3, 6 to 9)
│   └── E2E_AUDIT.json              # seeds, row counts and SHA-256 per result file
├── reproduce/                      # verify_paper_numbers.py, expected_numbers.csv, verify_against.sh
├── docs/                           # rerun details, LSTF dataset setup, recorded asset revisions
├── examples/                       # original cluster launchers (optional)
├── tests/                          # unit tests
├── assets/                         # README figures
├── THIRD_PARTY_NOTICES.md          # attribution for material adapted from Datadog Toto
├── licenses/                       # Apache-2.0 text and Datadog Toto NOTICE
└── requirements.txt, pyproject.toml
```
</details>

## Citation

```bibtex
@inproceedings{mavani2026observability,
  title         = {What Does an Observability Foundation Model Know?},
  author        = {Mavani, Dhyey Dharmendrakumar and Atri, Rian and Ji, Tairan},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2610.05577},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.05577}
}
```

## License

Our code, documentation and aggregate results are released under the [MIT License](LICENSE). Two files adapted from [Datadog Toto](https://github.com/DataDog/toto), `toto_interp/fev_tasks.py` and `toto_interp/boom.py`, keep their Apache-2.0 terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/). The MIT License doesn't cover third-party datasets, model weights or dependencies.
