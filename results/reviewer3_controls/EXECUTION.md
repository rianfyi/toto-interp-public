# Taxonomy-control suite: execution and compute record

The five-resplit inputs were produced under the launchers' default run root
(`runs/reviewer3_evalmode_20260728`, on cluster scratch). Only seeds 42--46
from the successful jobs described below enter the compact results in this
directory and `results/E2E_AUDIT.json`. Activation tensors and estimator
artifacts are not distributed.

## Protocol

- Each seed defines a series-disjoint train/validation/test resplit.
- Activation metadata and SHA256 hashes must match across pretrained,
  random-init, layer-permuted, and raw-control inputs.
- Representation views are selected on validation data and evaluated on test
  once.
- Repeated windows are averaged within series for conditional probes.
- Held-out rotations remove a domain or taxonomy combination from both
  training and validation.

## Eligible jobs and right-sized launchers

| Stage | Allocation used | Observed elapsed / peak RSS | Checked-in request |
|---|---|---|---|
| Taxonomy-control parent, seeds 42--46 | 16 CPU, 96 GB | 1:06:54--1:24:53 / 18.7--21.3 GB | 16 CPU, 32 GB |
| Consolidated held-out + conditional tail, seeds 42--46 | 2 CPU, 24 GB | 11:34--21:25 / up to 8.8 GB | 2 CPU, 16 GB |
| Layer-permuted probes, seeds 42--46 | -- | 38:47--1:19:01 / about 79 GB | 96 GB |
| MOMENT random-init CPU probe tail | 120 GB | completed / about 91 GB | 100 GB |

The parent estimator grid saturated roughly 15--16 cores, while the tail
saturated approximately two. The checked-in parent releases its 16-core
allocation after the unconditional controls. One tail task per seed then loads
each pretrained/random activation source once and evaluates all four held-out
cells plus conditional probes in-process. This replaces 15 separately queued
tasks with five and avoids repeated activation loads. Slurm `aftercorr`
dependencies pair each tail seed with its own successful parent seed, so one
failed seed cannot create an impossible all-array dependency.

The checked-in A10 launchers also stop after GPU-dependent activation or
interchange work. MOMENT taxonomy/dynamic fitting and provenance enrichment
run in correlated CPU follow-ups; the taxonomy-control A10 stage requests
32 GB for two hours and hands its CPU work to the 16-core parent above.

GPU-dependent activation or intervention stages (MOMENT structural, Toto
matched patch, MOMENT random activations, layer-permuted activations,
transfer, and MOMENT interchange) used NVIDIA A10 nodes. The MOMENT random
activation job was stopped only after all required activation files were
verified readable; its completed files fed the eligible CPU tail.

| A10 stage | Observed elapsed / peak RSS | Checked-in request |
|---|---:|---:|
| MOMENT pretrained activation + historical inline probe | 21:21--37:17 / 28.6--29.1 GB | activation only, 40 GB, 1 h |
| Toto matched patch | 3:19--3:56 / 6.7--9.0 GB | 24 GB, 30 min |
| MOMENT random activation + interrupted inline probe | 48:03--53:06 / 29.3--30.9 GB | activation only, 40 GB, 2 h |
| Layer-permuted activation | 6:40--7:27 / 42.2--49.2 GB | 64 GB, 1 h |
| Fixed transfer | 0:37--0:58 / 1.4--1.8 GB | 8 GB, 30 min |
| MOMENT matched interchange | 0:59--1:36 / 1.7--3.0 GB | 8 GB, 30 min |

Canceled, stale, duplicate, wrong-root, or superseded jobs are not read by the
aggregators.

## Audit and statistical scope

The taxonomy-control summarizer enforces exact activation provenance, matching
split hashes, and zero train/validation/test series overlap before producing
these aggregates. `results/E2E_AUDIT.json` then requires exactly seeds 42--46
once per expected aggregate cell, finite headline metrics, exact row counts,
and aggregate file hashes. It records 80 marginal rows, 40 unconditional rows,
20 layer-permuted rows, 30 conditional rows, 60 confound rows, and 40 held-out
rows.

Intervals are Student-\(t_4\) 95% interval half-widths across the five
resplits. They summarize between-resplit variability within BOOM. Paired win
counts are directional-consistency summaries, not significance claims.
Held-out brackets are coverage ranges because each rotation selects a
different held-out value.
