"""
Toto donor exchange for future_burstiness.

The donor exchange tests forecast behavior by blending layer-K residuals from
real donor series into a low-future-burstiness target's state, comparing a
high-burst donor with a randomized donor on the same target:

    1. Bin windows of --split (the reported runs use held-out test windows)
       by future_burstiness into HIGH (top quartile) and LOW (bottom
       quartile) groups.
    2. For each of up to --num-pairs low targets sampled per seed, draw a
       high-burst donor and a randomized donor from other series, and run
       three forecasts on the target's context:
         (a) clean        - no exchange
         (b) real-patched - the target's layer-K residuals at the patched
                            token position blended toward the high-burst
                            donor's (--blend; 1.0 replaces them)
         (c) null-patched - the same blend toward a randomized donor (any
                            eligible window from another series)
    3. Record WAPE, MASE, forecast burstiness, and the probe score for each
       condition.

Donor residuals come from real forward passes rather than synthetic
directions. Both arms use the same target; the randomized donor is not
covariate- or taxonomy-matched to the high-burst donor.
summarize_moment_exchange_transfer.py reports the probe check (fraction of
targets whose probe score is higher under the high-burst donor than under the
randomized donor), the forecast endpoint (fraction of targets whose forecast is
burstier under the high-burst donor than under the randomized donor), and the
median WAPE ratio (high-burst/randomized) as a secondary endpoint.

Key columns of paired_patch_results.csv:
  pair_id, seed, target_window_id, source_window_id, condition,
  wape, mase, probe_score, forecast_burstiness, wape_ratio_vs_clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.bootstrap import ensure_toto_importable

ensure_toto_importable()

from toto.inference.forecaster import TotoForecaster
from toto_interp import (
    PairedPatchConfig,
    ProbeArtifact,
    TraceConfig,
    apply_paired_patch,
    capture_source,
    extract_activations,
)
from toto_interp.boom import (
    build_boom_windows,
    build_masked_timeseries,
    ensure_boom_snapshot,
    load_boom_taxonomy,
    split_boom_series_ids,
)
from toto_interp.labels import robust_scale
from toto_interp.loader import load_toto_with_fallback, resolve_device
from toto_interp.metrics import mase, wape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-path", type=Path, required=True,
                   help="Path to fitted future_burstiness probe (.pt). Layer/token "
                        "selection inherits from the probe.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="BOOM train/validation/test resplit seed. Defaults to the first "
             "pair-sampling seed for backward compatibility.",
    )
    p.add_argument("--context-length", type=int, default=512,
                   help="Context length in observations. The default of 512 "
                        "reduces memory use; the reported runs use 1024.")
    p.add_argument("--max-series", type=int, default=200)
    p.add_argument("--max-windows-per-series", type=int, default=4)
    p.add_argument("--num-pairs", type=int, default=40,
                   help="Number of (low-target, high-source) pairs per seed.")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--snapshot-path", type=Path, default=None)
    p.add_argument("--high-quantile", type=float, default=0.75)
    p.add_argument("--low-quantile", type=float, default=0.25)
    p.add_argument("--patch-layer", type=int, default=None,
                   help="Override layer to patch (default: probe.layer).")
    p.add_argument("--patch-token-position", type=str, default=None,
                   help="Override token position (default: probe.token_position).")
    p.add_argument("--blend", type=float, default=1.0,
                   help="Patch blend: 1.0 = full replace, 0.0 = no-op.")
    p.add_argument("--max-windows-eval", type=int, default=600,
                   help="Cap on total windows considered for binning.")
    return p.parse_args()


def forecast_window(
    forecaster: TotoForecaster,
    window,
    *,
    num_samples: int,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (median, samples) on CPU."""
    masked = build_masked_timeseries(window.context).to(forecaster.model.device)
    samples_per_batch = min(num_samples, 8)
    forecast = forecaster.forecast(
        inputs=masked,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        use_kv_cache=True,
    )
    return forecast.median.cpu(), forecast.samples.cpu()


def forecast_burstiness(forecast_median: torch.Tensor, context: torch.Tensor) -> float:
    """
    Match the labels.py future_burstiness definition: max scaled deviation of
    forecast vs. context's robust scale.
    """
    # forecast_median shape: (variates, prediction_length); context shape: (variates, T)
    scale = robust_scale(context)  # (variates,)
    if not torch.is_tensor(scale):
        scale = torch.as_tensor(scale)
    median_ctx = context.median(dim=-1, keepdim=True).values
    deviation = (forecast_median - median_ctx).abs()
    scale_b = scale.view(-1, 1).clamp_min(1e-6)
    scaled = deviation / scale_b
    return float(scaled.max().item())


def probe_score(activations: torch.Tensor, probe: ProbeArtifact) -> float:
    if probe.feature_mean is None or probe.feature_std is None:
        return float("nan")
    standardized = (activations - probe.feature_mean) / probe.feature_std
    standardized = torch.nan_to_num(standardized, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
    score = standardized @ probe.coef[0].unsqueeze(-1)
    return float((score.squeeze(-1) + probe.intercept[0]).mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    model,
    forecaster: TotoForecaster,
    probe: ProbeArtifact,
    backbone,
    patch_size: int,
    patch_layer: int,
    patch_token: str,
    trace_config: TraceConfig,
    windows_all,
    windows_high,
    windows_low,
    windows_pool,
) -> list[dict]:
    rng = np.random.default_rng(seed)

    # Sample targets without replacement. Sources are also unique within each
    # arm and must come from a different series than the target; the null source
    # must additionally differ from the real-source series.
    n_pairs = min(args.num_pairs, len(windows_high), len(windows_low))
    low_idx = rng.choice(len(windows_low), size=n_pairs, replace=False)
    used_real_window_ids: set[str] = set()
    used_null_window_ids: set[str] = set()

    rows: list[dict] = []
    for pair_id, l_i in enumerate(low_idx):
        target = windows_low[l_i]
        real_candidates = [
            window
            for window in windows_high
            if window.series_id != target.series_id
            and window.window_id not in used_real_window_ids
        ]
        if not real_candidates:
            logger.warning("pair %d: no unused cross-series real source; stopping", pair_id)
            break
        source_real = real_candidates[int(rng.integers(0, len(real_candidates)))]
        used_real_window_ids.add(source_real.window_id)

        null_candidates = [
            window
            for window in windows_pool
            if window.series_id not in {target.series_id, source_real.series_id}
            and window.window_id != source_real.window_id
            and window.window_id not in used_null_window_ids
        ]
        if not null_candidates:
            logger.warning("pair %d: no unused cross-series null source; stopping", pair_id)
            break
        source_null = null_candidates[int(rng.integers(0, len(null_candidates)))]
        used_null_window_ids.add(source_null.window_id)

        # ---- Capture source residuals (real + null) ----
        with capture_source(backbone, (patch_layer,)) as cap_real:
            _ = forecast_window(forecaster, source_real,
                                num_samples=args.num_samples,
                                prediction_length=patch_size)
        with capture_source(backbone, (patch_layer,)) as cap_null:
            _ = forecast_window(forecaster, source_null,
                                num_samples=args.num_samples,
                                prediction_length=patch_size)

        if patch_layer not in cap_real or patch_layer not in cap_null:
            logger.warning("pair %d: capture missed layer %d; skipping", pair_id, patch_layer)
            continue

        # Use common forecast random numbers across clean, real, and null
        # conditions so the paired contrast is not inflated by sampling noise.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        def restore_forecast_rng() -> None:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        # ---- Clean target ----
        restore_forecast_rng()
        clean_median, _ = forecast_window(forecaster, target,
                                          num_samples=args.num_samples,
                                          prediction_length=patch_size)
        clean_burst = forecast_burstiness(clean_median.squeeze(0), target.context)
        clean_wape = wape(target.next_patch, clean_median.squeeze(0))
        clean_mase = mase(target.context, target.next_patch, clean_median.squeeze(0))

        clean_act_batch = extract_activations(model, [target], trace_config)
        clean_score = probe_score(clean_act_batch.activations, probe)

        # ---- Real patch (high source -> low target) ----
        real_cfg = PairedPatchConfig(
            layer_indices=(patch_layer,),
            token_position=patch_token,
            source_residuals={patch_layer: cap_real[patch_layer]},
            blend=args.blend,
        )
        restore_forecast_rng()
        with apply_paired_patch(backbone, real_cfg):
            real_median, _ = forecast_window(forecaster, target,
                                              num_samples=args.num_samples,
                                              prediction_length=patch_size)
        real_burst = forecast_burstiness(real_median.squeeze(0), target.context)
        real_wape = wape(target.next_patch, real_median.squeeze(0))
        real_mase = mase(target.context, target.next_patch, real_median.squeeze(0))
        with apply_paired_patch(backbone, real_cfg):
            real_act_batch = extract_activations(model, [target], trace_config)
        real_score = probe_score(real_act_batch.activations, probe)

        # ---- Null patch (random source -> low target) ----
        null_cfg = PairedPatchConfig(
            layer_indices=(patch_layer,),
            token_position=patch_token,
            source_residuals={patch_layer: cap_null[patch_layer]},
            blend=args.blend,
        )
        restore_forecast_rng()
        with apply_paired_patch(backbone, null_cfg):
            null_median, _ = forecast_window(forecaster, target,
                                              num_samples=args.num_samples,
                                              prediction_length=patch_size)
        null_burst = forecast_burstiness(null_median.squeeze(0), target.context)
        null_wape = wape(target.next_patch, null_median.squeeze(0))
        null_mase = mase(target.context, target.next_patch, null_median.squeeze(0))
        with apply_paired_patch(backbone, null_cfg):
            null_act_batch = extract_activations(model, [target], trace_config)
        null_score = probe_score(null_act_batch.activations, probe)

        target_label = float(target.labels.get("future_burstiness", float("nan")))
        source_real_label = float(source_real.labels.get("future_burstiness", float("nan")))
        source_null_label = float(source_null.labels.get("future_burstiness", float("nan")))

        for cond, m_wape, m_mase, m_burst, m_score, src_id, src_label in [
            ("clean", clean_wape, clean_mase, clean_burst, clean_score, "(none)", float("nan")),
            ("real_patch", real_wape, real_mase, real_burst, real_score, source_real.window_id, source_real_label),
            ("null_patch", null_wape, null_mase, null_burst, null_score, source_null.window_id, source_null_label),
        ]:
            rows.append({
                "seed": seed,
                "split_seed": args.split_seed if args.split_seed is not None else args.seeds[0],
                "pair_id": pair_id,
                "target_series_id": target.series_id,
                "target_window_id": target.window_id,
                "target_future_burstiness_label": target_label,
                "source_window_id": src_id,
                "source_series_id": (
                    "(none)"
                    if cond == "clean"
                    else source_real.series_id
                    if cond == "real_patch"
                    else source_null.series_id
                ),
                "source_future_burstiness_label": src_label,
                "condition": cond,
                "patch_layer": patch_layer,
                "patch_token_position": patch_token,
                "wape": float(m_wape),
                "mase": float(m_mase),
                "probe_score": m_score,
                "forecast_burstiness": m_burst,
                "wape_ratio_vs_clean": (float(m_wape) / clean_wape) if clean_wape > 0 else float("nan"),
            })

        if (pair_id + 1) % 5 == 0:
            logger.info("seed %d: completed %d/%d pairs", seed, pair_id + 1, n_pairs)

    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe = ProbeArtifact.load(args.probe_path)
    probe_sha256 = hashlib.sha256(args.probe_path.read_bytes()).hexdigest()
    if probe.label_spec.name != "future_burstiness":
        logger.warning("Probe label is %r, expected future_burstiness", probe.label_spec.name)

    patch_layer = args.patch_layer if args.patch_layer is not None else probe.layer
    patch_token = args.patch_token_position or probe.token_position
    logger.info("Patch site: layer=%d token=%s blend=%.2f", patch_layer, patch_token, args.blend)

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    model = load_toto_with_fallback(args.model_id, map_location="cpu", device=device)
    if model.training:
        raise RuntimeError("Toto must be in evaluation mode during paired patching.")
    backbone = model.model
    forecaster = TotoForecaster(backbone)
    patch_size = int(backbone.patch_embed.patch_size)
    if args.context_length % patch_size != 0:
        raise ValueError(f"context-length must be divisible by patch size {patch_size}")

    # Build BOOM windows on the chosen split
    taxonomy = load_boom_taxonomy(local_path=args.snapshot_path)
    split_seed = args.split_seed if args.split_seed is not None else args.seeds[0]
    split_ids = split_boom_series_ids(taxonomy, seed=split_seed)
    if args.max_series > 0:
        split_ids = {
            name: ids[: args.max_series]
            for name, ids in split_ids.items()
        }
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    split_overlap_counts = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(split_overlap_counts.values()):
        raise RuntimeError(f"BOOM series splits overlap: {split_overlap_counts}")
    split_manifest = {
        name: {
            "series_count": len(ids),
            "series_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
        for name, ids in split_ids.items()
    }
    series_ids = split_ids[args.split]
    series_id_sha256 = hashlib.sha256("\n".join(series_ids).encode()).hexdigest()
    snapshot_path = ensure_boom_snapshot(series_ids, local_path=args.snapshot_path)
    windows = build_boom_windows(
        series_ids=series_ids,
        split=args.split,
        snapshot_path=snapshot_path,
        taxonomy=taxonomy,
        context_length=args.context_length,
        patch_size=patch_size,
        max_windows_per_series=args.max_windows_per_series,
        include_heldout_late=False,
    )
    if args.max_windows_eval is not None and len(windows) > args.max_windows_eval:
        windows = windows[: args.max_windows_eval]
    logger.info("Loaded %d windows from split=%s", len(windows), args.split)

    # Bin by future_burstiness label
    label_values = np.array([float(w.labels.get("future_burstiness", float("nan"))) for w in windows])
    valid = ~np.isnan(label_values)
    if valid.sum() < 20:
        raise RuntimeError(f"Too few valid future_burstiness labels: {valid.sum()}")
    high_thr = np.quantile(label_values[valid], args.high_quantile)
    low_thr = np.quantile(label_values[valid], args.low_quantile)
    windows_high = [w for w, v in zip(windows, label_values) if not np.isnan(v) and v >= high_thr]
    windows_low = [w for w, v in zip(windows, label_values) if not np.isnan(v) and v <= low_thr]
    windows_pool = [w for w, v in zip(windows, label_values) if not np.isnan(v)]
    logger.info("Bins: high=%d (>=%.3g) low=%d (<=%.3g) pool=%d",
                len(windows_high), high_thr, len(windows_low), low_thr, len(windows_pool))

    trace_config = TraceConfig(
        layers=(patch_layer,),
        token_positions=(patch_token,),
        pooling_modes=(probe.pooling_mode,),
        capture_patch_embedding=False,
        use_kv_cache=True,
    )

    all_rows: list[dict] = []
    for seed in args.seeds:
        logger.info("=== seed %d ===", seed)
        torch.manual_seed(seed)
        rows = run_seed(
            seed=seed,
            args=args,
            model=model,
            forecaster=forecaster,
            probe=probe,
            backbone=backbone,
            patch_size=patch_size,
            patch_layer=patch_layer,
            patch_token=patch_token,
            trace_config=trace_config,
            windows_all=windows,
            windows_high=windows_high,
            windows_low=windows_low,
            windows_pool=windows_pool,
        )
        all_rows.extend(rows)
        # Snapshot CSV after each seed in case the run is interrupted
        pd.DataFrame(all_rows).to_csv(args.output_dir / "paired_patch_results.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "paired_patch_results.csv", index=False)

    # Aggregate summary
    summary = (
        df.groupby(["condition", "seed"], dropna=False)
        .agg(
            wape_mean=("wape", "mean"),
            wape_median=("wape", "median"),
            wape_ratio_mean=("wape_ratio_vs_clean", "mean"),
            wape_ratio_median=("wape_ratio_vs_clean", "median"),
            mase_mean=("mase", "mean"),
            probe_score_mean=("probe_score", "mean"),
            forecast_burstiness_mean=("forecast_burstiness", "mean"),
            forecast_burstiness_median=("forecast_burstiness", "median"),
            n=("pair_id", "count"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "paired_patch_summary.csv", index=False)

    with open(args.output_dir / "paired_patch_meta.json", "w") as fh:
        json.dump(
            {
                "probe_path": str(args.probe_path),
                "probe_sha256": probe_sha256,
                "probe_label": probe.label_spec.name,
                "probe_seed": probe.seed,
                "probe_model_id": probe.model_id,
                "probe_weight_source": probe.weight_source,
                "probe_activation_source_signature": probe.artifact_metadata.get(
                    "activation_source_signature"
                ),
                "probe_layer": probe.layer,
                "probe_token_position": probe.token_position,
                "probe_pooling_mode": probe.pooling_mode,
                "patch_layer": patch_layer,
                "patch_token_position": patch_token,
                "blend": args.blend,
                "device": device,
                "model_training": bool(model.training),
                "context_length": args.context_length,
                "split": args.split,
                "split_seed": split_seed,
                "series_count": len(series_ids),
                "series_id_sha256": series_id_sha256,
                "split_manifest": split_manifest,
                "split_overlap_counts": split_overlap_counts,
                "num_pairs_per_seed": args.num_pairs,
                "num_samples": args.num_samples,
                "seeds": args.seeds,
                "common_forecast_random_numbers": True,
                "cross_series_sources": True,
                "high_quantile": args.high_quantile,
                "low_quantile": args.low_quantile,
                "high_threshold": float(high_thr),
                "low_threshold": float(low_thr),
                "n_windows": len(windows),
                "n_high": len(windows_high),
                "n_low": len(windows_low),
            },
            fh,
            indent=2,
        )

    logger.info("Done. %d rows -> %s", len(df), args.output_dir / "paired_patch_results.csv")


if __name__ == "__main__":
    main()
