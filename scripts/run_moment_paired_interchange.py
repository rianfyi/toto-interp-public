"""
Matched source-residual interchange for MOMENT-base future burstiness.

The protocol is intentionally paired and head-agnostic:

* The probe view is either selected once by validation R2 from the existing
  five-resplit MOMENT probe grid or supplied as a predeclared artifact.
* Held-out low-future-burstiness targets are shared by the real and null arms.
* A real donor comes from the high-future-burstiness quartile. Its null donor
  is randomized among the nearest context-matched, non-high donors.
* Every donor is from a different series and has exactly the target's channel
  count; no synthetic channel broadcasting is permitted.
* The exact same pair triples are evaluated at blends 0.25, 0.5, and 1.0.

The primary endpoint is the fitted future-burstiness probe score at the fixed
MOMENT view (real minus null within target). When the installed official
MOMENT API exposes its pretrained reconstruction head and ``short_forecast``,
the runner also reports held-out next-patch error as a secondary behavioral
analogue. If that API is unavailable under ``--secondary-endpoint auto``, the
secondary endpoint is explicitly marked unavailable rather than synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.boom import (
    build_boom_windows,
    ensure_boom_snapshot,
    load_boom_taxonomy,
    raw_features_for_window,
    split_boom_series_ids,
)
from toto_interp.labels import robust_scale
from toto_interp.loader import resolve_device
from toto_interp.moment_interchange import (
    MomentInterchangeConfig,
    apply_moment_interchange,
    capture_moment_residual,
    score_moment_probe,
)
from toto_interp.moment_loader import (
    MOMENT_BASE_ID,
    load_moment_with_fallback,
    moment_hidden_dim,
    moment_num_layers,
    moment_patch_size,
)
from toto_interp.types import ProbeArtifact, WindowExample

logger = logging.getLogger(__name__)

PRIMARY_LABEL = "future_burstiness"
DEFAULT_BLENDS = (0.25, 0.5, 1.0)
MATCH_CATEGORICAL_FIELDS = ("frequency_bucket", "metric_type", "domain")


@dataclass(frozen=True)
class ProbeSelection:
    path: Path
    rule: str
    validation_metric: str | None
    validation_score: float | None
    result_row: dict[str, Any] | None


@dataclass(frozen=True)
class PairTriple:
    pair_id: int
    target: WindowExample
    real_source: WindowExample
    null_source: WindowExample
    null_match_distance: float
    null_match_rank: int
    null_candidate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    probe_group = parser.add_mutually_exclusive_group(required=True)
    probe_group.add_argument(
        "--probe-results",
        type=Path,
        help="Existing per-resplit MOMENT dynamic probe_results.csv. The view is "
        "selected by validation R2 only.",
    )
    probe_group.add_argument(
        "--probe-path",
        type=Path,
        help="Predeclared future_burstiness MOMENT probe artifact.",
    )
    parser.add_argument("--activation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=MOMENT_BASE_ID)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Pair randomization seed. Defaults to --split-seed.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--num-pairs", type=int, default=40)
    parser.add_argument("--blends", type=float, nargs="+", default=list(DEFAULT_BLENDS))
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument(
        "--null-match-k",
        type=int,
        default=5,
        help="Randomize the null donor among the K nearest eligible context matches.",
    )
    parser.add_argument(
        "--secondary-endpoint",
        choices=("auto", "required", "off"),
        default="auto",
        help="Use MOMENT's official short-forecast reconstruction head when available.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def series_id_sha256(series_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(series_ids).encode()).hexdigest()


def select_validation_probe_row(frame: pd.DataFrame) -> pd.Series:
    """Choose the future-burstiness view without consulting test performance."""

    required = {
        "label",
        "layer",
        "token_position",
        "pooling_mode",
        "artifact_path",
        "val_r2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Probe results are missing required columns: {sorted(missing)}")

    candidates = frame[frame["label"].astype(str) == PRIMARY_LABEL].copy()
    if "method" in candidates.columns:
        candidates = candidates[candidates["method"].astype(str) == "linear_probe"]
    candidates["_val_r2_numeric"] = pd.to_numeric(candidates["val_r2"], errors="coerce")
    candidates = candidates.dropna(subset=["_val_r2_numeric"])
    if candidates.empty:
        raise ValueError(
            f"No finite validation-R2 linear-probe rows found for {PRIMARY_LABEL!r}."
        )

    # Test columns are deliberately absent from both the ordering and tie-break.
    candidates["_layer_numeric"] = pd.to_numeric(candidates["layer"], errors="raise")
    selected = candidates.sort_values(
        ["_val_r2_numeric", "_layer_numeric", "token_position", "pooling_mode", "artifact_path"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).iloc[0]
    return selected.drop(labels=["_val_r2_numeric", "_layer_numeric"])


def _resolve_artifact_path(raw_path: object, results_path: Path) -> Path:
    candidate = Path(str(raw_path))
    candidates = [
        candidate,
        results_path.parent / candidate,
        results_path.parent / "artifacts" / candidate.name,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Selected probe artifact {raw_path!r} does not exist. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_probe_selection(args: argparse.Namespace) -> ProbeSelection:
    if args.probe_results is not None:
        frame = pd.read_csv(args.probe_results)
        selected = select_validation_probe_row(frame)
        path = _resolve_artifact_path(selected["artifact_path"], args.probe_results)
        return ProbeSelection(
            path=path,
            rule="validation_selected",
            validation_metric="val_r2",
            validation_score=float(selected["val_r2"]),
            result_row={
                key: (None if pd.isna(value) else value)
                for key, value in selected.to_dict().items()
            },
        )

    path = args.probe_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Predeclared probe artifact does not exist: {path}")
    return ProbeSelection(
        path=path,
        rule="predeclared_artifact",
        validation_metric=None,
        validation_score=None,
        result_row=None,
    )


def _summary_split_manifest(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = summary.get("split_manifest")
    if isinstance(manifest, dict):
        return manifest
    splits = summary.get("splits")
    if isinstance(splits, dict):
        return {
            name: {
                "series_count": values.get("series_count"),
                "series_id_sha256": values.get("series_id_sha256"),
            }
            for name, values in splits.items()
        }
    raise ValueError("Activation summary has no split manifest.")


def validate_activation_summary(
    summary: dict[str, Any],
    *,
    split_seed: int,
    model_id: str,
) -> None:
    if int(summary.get("seed", -1)) != split_seed:
        raise ValueError(
            f"Activation summary seed={summary.get('seed')} does not match split seed {split_seed}."
        )
    if str(summary.get("model_id")) != model_id:
        raise ValueError(
            f"Activation model {summary.get('model_id')!r} does not match {model_id!r}."
        )
    if str(summary.get("weight_source")) != "pretrained":
        raise ValueError("MOMENT interchange requires pretrained probe activations.")
    if summary.get("model_training") is not False:
        raise ValueError("Activation summary must record model_training=false.")
    overlap = summary.get("split_overlap_counts")
    overlap_keys = {"train_val", "train_test", "val_test"}
    if (
        not isinstance(overlap, dict)
        or not overlap_keys.issubset(overlap)
        or any(int(overlap[key]) for key in overlap_keys)
    ):
        raise ValueError("Activation summary does not establish series-disjoint train/val/test splits.")
    for split_name in ("train", "val", "test"):
        values = _summary_split_manifest(summary).get(split_name)
        if not values or not values.get("series_id_sha256"):
            raise ValueError(f"Activation summary is missing {split_name} series provenance.")


def _parse_probe_signature(probe: ProbeArtifact) -> tuple[str, dict[str, Any]]:
    signature_raw = probe.artifact_metadata.get("activation_source_signature")
    try:
        signature = json.loads(signature_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Probe artifact has no parseable activation source signature.") from exc
    if not isinstance(signature, dict):
        raise ValueError("Probe activation source signature must decode to an object.")
    return str(signature_raw), signature


def validate_probe_provenance(
    probe: ProbeArtifact,
    *,
    selection: ProbeSelection,
    summary: dict[str, Any],
    split_seed: int,
    model_id: str,
    split_manifest: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if probe.label_spec.name != PRIMARY_LABEL or probe.label_spec.task_type != "continuous":
        raise ValueError(
            f"Expected a continuous {PRIMARY_LABEL!r} probe, received "
            f"{probe.label_spec.name!r}/{probe.label_spec.task_type!r}."
        )
    if probe.method != "linear_probe":
        raise ValueError("MOMENT interchange requires a fitted linear probe artifact.")
    if probe.model_id != model_id or probe.weight_source != "pretrained":
        raise ValueError(
            "Probe/model provenance mismatch: "
            f"probe={probe.model_id}/{probe.weight_source}, run={model_id}/pretrained."
        )
    if int(probe.seed) != split_seed:
        raise ValueError(f"Probe seed={probe.seed} does not match split seed {split_seed}.")
    if probe.token_position not in ("all_context", "final_context"):
        raise ValueError(
            f"MOMENT interchange cannot use token position {probe.token_position!r}."
        )
    if probe.pooling_mode not in ("series_mean", "per_variate"):
        raise ValueError(f"Unsupported MOMENT pooling mode {probe.pooling_mode!r}.")

    if selection.result_row is not None:
        row = selection.result_row
        expected = {
            "layer": int(probe.layer),
            "token_position": probe.token_position,
            "pooling_mode": probe.pooling_mode,
        }
        observed = {
            "layer": int(row["layer"]),
            "token_position": str(row["token_position"]),
            "pooling_mode": str(row["pooling_mode"]),
        }
        if observed != expected:
            raise ValueError(
                f"Validation-selected probe row {observed} disagrees with artifact {expected}."
            )
        if "seed" in row and row["seed"] is not None and int(row["seed"]) != split_seed:
            raise ValueError("Validation-selected probe result row has the wrong resplit seed.")

    signature_raw, signature = _parse_probe_signature(probe)
    required_signature = {
        "seed": split_seed,
        "model_id": model_id,
        "weight_source": "pretrained",
        "context_length": int(summary["context_length"]),
        "patch_size": int(summary["patch_size"]),
        "model_training": False,
    }
    for key, expected in required_signature.items():
        if signature.get(key) != expected:
            raise ValueError(
                f"Probe activation signature {key}={signature.get(key)!r}, expected {expected!r}."
            )

    signature_hashes = signature.get("series_id_sha256_by_split")
    if not isinstance(signature_hashes, dict):
        raise ValueError("Probe activation signature has no per-split series hashes.")
    for split_name, values in split_manifest.items():
        expected_hash = str(values["series_id_sha256"])
        if signature_hashes.get(split_name) != expected_hash:
            raise ValueError(
                f"Probe activation signature disagrees on the {split_name} series split."
            )
    return signature_raw, signature


def reconstruct_and_validate_splits(
    *,
    taxonomy: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    split_seed: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], dict[str, int]]:
    split_ids = split_boom_series_ids(taxonomy, seed=split_seed)
    max_series = int(summary.get("max_series_per_split", 0))
    if max_series > 0:
        split_ids = {name: ids[:max_series] for name, ids in split_ids.items()}

    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    overlap = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Reconstructed BOOM splits overlap: {overlap}")

    manifest = {
        name: {
            "series_count": len(ids),
            "series_id_sha256": series_id_sha256(ids),
        }
        for name, ids in split_ids.items()
    }
    recorded = _summary_split_manifest(summary)
    for split_name, values in manifest.items():
        recorded_values = recorded.get(split_name, {})
        if int(recorded_values.get("series_count", -1)) != int(values["series_count"]):
            raise ValueError(f"Activation/intervention {split_name} series counts disagree.")
        if str(recorded_values.get("series_id_sha256")) != str(values["series_id_sha256"]):
            raise ValueError(f"Activation/intervention {split_name} series hashes disagree.")
    return split_ids, manifest, overlap


def _future_burstiness(window: WindowExample) -> float:
    return float(window.labels.get(PRIMARY_LABEL, float("nan")))


def _match_feature_matrix(windows: list[WindowExample]) -> tuple[np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    index: dict[str, int] = {}
    for row_idx, window in enumerate(windows):
        raw = raw_features_for_window(window).cpu().numpy().astype(np.float64)
        dynamic = np.asarray(
            [
                float(window.labels.get("current_sparsity", 0.0)),
                float(window.labels.get("current_burstiness", 0.0)),
            ],
            dtype=np.float64,
        )
        rows.append(np.concatenate([raw, dynamic]))
        index[window.window_id] = row_idx
    matrix = np.stack(rows)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = np.nan_to_num((matrix - mean) / std, nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(standardized, -1e3, 1e3), index


def _null_match_distance(
    real: WindowExample,
    candidate: WindowExample,
    *,
    features: np.ndarray,
    feature_index: dict[str, int],
) -> float:
    real_features = features[feature_index[real.window_id]]
    candidate_features = features[feature_index[candidate.window_id]]
    continuous = float(
        np.linalg.norm(real_features - candidate_features)
        / math.sqrt(max(1, real_features.size))
    )
    categorical = sum(
        str(real.labels.get(field)) != str(candidate.labels.get(field))
        for field in MATCH_CATEGORICAL_FIELDS
    )
    return continuous + float(categorical)


def build_matched_pair_triples(
    windows: list[WindowExample],
    *,
    num_pairs: int,
    seed: int,
    high_quantile: float,
    low_quantile: float,
    null_match_k: int,
) -> tuple[list[PairTriple], float, float]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    if null_match_k <= 0:
        raise ValueError("null_match_k must be positive.")
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("Expected 0 < low_quantile < high_quantile < 1.")

    values = np.asarray([_future_burstiness(window) for window in windows], dtype=np.float64)
    finite = np.isfinite(values)
    if int(finite.sum()) < max(20, num_pairs):
        raise ValueError(f"Too few finite future-burstiness windows: {int(finite.sum())}.")
    high_threshold = float(np.quantile(values[finite], high_quantile))
    low_threshold = float(np.quantile(values[finite], low_quantile))

    valid_windows = [window for window, keep in zip(windows, finite.tolist()) if keep]
    targets = [
        window for window in valid_windows if _future_burstiness(window) <= low_threshold
    ]
    real_pool = [
        window for window in valid_windows if _future_burstiness(window) >= high_threshold
    ]
    null_pool = [
        window for window in valid_windows if _future_burstiness(window) < high_threshold
    ]
    features, feature_index = _match_feature_matrix(valid_windows)

    rng = np.random.default_rng(seed)
    target_order = rng.permutation(len(targets)).tolist()
    used_window_ids: set[str] = set()
    used_target_series: set[str] = set()
    triples: list[PairTriple] = []

    for target_index in target_order:
        if len(triples) >= num_pairs:
            break
        target = targets[target_index]
        if target.series_id in used_target_series or target.window_id in used_window_ids:
            continue
        num_variates = int(target.context.shape[0])
        real_candidates = [
            source
            for source in real_pool
            if source.series_id != target.series_id
            and source.window_id not in used_window_ids
            and int(source.context.shape[0]) == num_variates
        ]
        if not real_candidates:
            continue

        real_order = rng.permutation(len(real_candidates)).tolist()
        selected: tuple[WindowExample, WindowExample, float, int, int] | None = None
        for real_index in real_order:
            real_source = real_candidates[real_index]
            null_candidates = [
                source
                for source in null_pool
                if source.series_id not in {target.series_id, real_source.series_id}
                and source.window_id not in used_window_ids
                and source.window_id != real_source.window_id
                and int(source.context.shape[0]) == num_variates
            ]
            if not null_candidates:
                continue

            ranked = sorted(
                (
                    (
                        _null_match_distance(
                            real_source,
                            source,
                            features=features,
                            feature_index=feature_index,
                        ),
                        source.window_id,
                        source,
                    )
                    for source in null_candidates
                ),
                key=lambda item: (item[0], item[1]),
            )
            nearest = ranked[: min(null_match_k, len(ranked))]
            chosen_index = int(rng.integers(0, len(nearest)))
            distance, _, null_source = nearest[chosen_index]
            selected = (
                real_source,
                null_source,
                float(distance),
                chosen_index + 1,
                len(null_candidates),
            )
            break

        if selected is None:
            continue
        real_source, null_source, distance, match_rank, candidate_count = selected
        triple = PairTriple(
            pair_id=len(triples),
            target=target,
            real_source=real_source,
            null_source=null_source,
            null_match_distance=distance,
            null_match_rank=match_rank,
            null_candidate_count=candidate_count,
        )
        triples.append(triple)
        used_window_ids.update(
            {
                target.window_id,
                real_source.window_id,
                null_source.window_id,
            }
        )
        used_target_series.add(target.series_id)

    if len(triples) != num_pairs:
        shape_counts = pd.Series(
            [int(window.context.shape[0]) for window in valid_windows]
        ).value_counts()
        raise RuntimeError(
            f"Could construct only {len(triples)}/{num_pairs} exact-channel matched triples. "
            f"Available window channel counts: {shape_counts.to_dict()}."
        )
    validate_pair_triples(triples, high_threshold=high_threshold, low_threshold=low_threshold)
    return triples, high_threshold, low_threshold


def validate_pair_triples(
    triples: list[PairTriple],
    *,
    high_threshold: float,
    low_threshold: float,
) -> None:
    exact = [
        (
            triple.target.window_id,
            triple.real_source.window_id,
            triple.null_source.window_id,
        )
        for triple in triples
    ]
    if len(set(exact)) != len(exact):
        raise ValueError("Exact target/real/null pair triples must be unique.")
    if len({triple.pair_id for triple in triples}) != len(triples):
        raise ValueError("pair_id values must be unique.")

    for triple in triples:
        series = {
            triple.target.series_id,
            triple.real_source.series_id,
            triple.null_source.series_id,
        }
        if len(series) != 3:
            raise ValueError(f"Pair {triple.pair_id} is not cross-series.")
        shapes = {
            tuple(triple.target.context.shape),
            tuple(triple.real_source.context.shape),
            tuple(triple.null_source.context.shape),
        }
        if len(shapes) != 1:
            raise ValueError(f"Pair {triple.pair_id} does not have exact context-shape matching.")
        if _future_burstiness(triple.target) > low_threshold:
            raise ValueError(f"Pair {triple.pair_id} target is outside the low stratum.")
        if _future_burstiness(triple.real_source) < high_threshold:
            raise ValueError(f"Pair {triple.pair_id} real donor is outside the high stratum.")
        if _future_burstiness(triple.null_source) >= high_threshold:
            raise ValueError(f"Pair {triple.pair_id} null donor is in the high stratum.")


def _moment_inputs(
    window: WindowExample,
    *,
    pipeline,
    device: str,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(window.context.shape)[-1] != seq_len:
        raise ValueError(
            f"Window {window.window_id} has context length {window.context.shape[-1]}, "
            f"expected {seq_len}."
        )
    dtype = next(pipeline.parameters()).dtype
    x_enc = window.context.unsqueeze(0).to(device=device, dtype=dtype)
    input_mask = torch.ones((1, seq_len), device=device, dtype=torch.float32)
    return x_enc, input_mask


@torch.inference_mode()
def _capture_forward(
    pipeline,
    *,
    layer: int,
    forward: Callable[[], Any],
) -> tuple[torch.Tensor, Any]:
    with capture_moment_residual(pipeline, layer=layer) as captured:
        output = forward()
    if "residual" not in captured:
        raise RuntimeError(f"MOMENT layer {layer} did not emit a residual.")
    return captured["residual"], output


def _embedding_forward(
    pipeline,
    window: WindowExample,
    *,
    device: str,
    seq_len: int,
) -> Callable[[], Any]:
    x_enc, input_mask = _moment_inputs(
        window, pipeline=pipeline, device=device, seq_len=seq_len
    )
    return lambda: pipeline.embed(
        x_enc=x_enc,
        input_mask=input_mask,
        reduction="none",
    )


def _short_forecast_forward(
    pipeline,
    window: WindowExample,
    *,
    device: str,
    seq_len: int,
    patch_size: int,
) -> Callable[[], Any]:
    x_enc, input_mask = _moment_inputs(
        window, pipeline=pipeline, device=device, seq_len=seq_len
    )
    return lambda: pipeline.short_forecast(
        x_enc=x_enc,
        input_mask=input_mask,
        forecast_horizon=patch_size,
    )


def _score_residual(
    residual: torch.Tensor,
    *,
    window: WindowExample,
    probe: ProbeArtifact,
) -> tuple[float, float, int]:
    return score_moment_probe(
        residual,
        num_variates=int(window.context.shape[0]),
        probe=probe,
    )


def _future_error_metrics(
    forecast_output: Any,
    *,
    window: WindowExample,
    patch_size: int,
) -> dict[str, float]:
    forecast = getattr(forecast_output, "forecast", None)
    if not isinstance(forecast, torch.Tensor):
        raise RuntimeError("MOMENT short_forecast did not return a tensor forecast.")
    prediction = forecast.detach().cpu().float()
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(f"Expected one target forecast, received {tuple(prediction.shape)}.")
        prediction = prediction[0]
    target = window.next_patch.detach().cpu().float()
    if prediction.shape[-1] != patch_size:
        raise ValueError(
            f"MOMENT forecast horizon {prediction.shape[-1]} does not match patch size {patch_size}."
        )
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"MOMENT forecast/target shapes disagree: {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )

    error = prediction - target
    scale = robust_scale(window.context.detach().cpu()).clamp_min(1e-6)
    scaled_absolute_error = error.abs() / scale
    return {
        "future_mae": float(error.abs().mean().item()),
        "future_mse": float(error.square().mean().item()),
        "future_scaled_mae": float(scaled_absolute_error.mean().item()),
    }


def _condition_row(
    *,
    triple: PairTriple,
    blend: float,
    condition: str,
    source: WindowExample | None,
    score: tuple[float, float, int],
    clean_score: float,
    future_metrics: dict[str, float] | None,
    clean_future_metrics: dict[str, float] | None,
    split_seed: int,
    sampling_seed: int,
    probe: ProbeArtifact,
) -> dict[str, Any]:
    source_label = float("nan") if source is None else _future_burstiness(source)
    row: dict[str, Any] = {
        "split_seed": split_seed,
        "seed": sampling_seed,
        "sampling_seed": sampling_seed,
        "pair_id": triple.pair_id,
        "blend": blend,
        "condition": condition,
        "target_series_id": triple.target.series_id,
        "target_window_id": triple.target.window_id,
        "target_future_burstiness_label": _future_burstiness(triple.target),
        "source_series_id": "(none)" if source is None else source.series_id,
        "source_window_id": "(none)" if source is None else source.window_id,
        "source_future_burstiness_label": source_label,
        "num_variates": int(triple.target.context.shape[0]),
        "null_match_distance": triple.null_match_distance,
        "null_match_rank": triple.null_match_rank,
        "null_candidate_count": triple.null_candidate_count,
        "patch_layer": int(probe.layer),
        "probe_layer": int(probe.layer),
        "probe_token_position": probe.token_position,
        "probe_pooling_mode": probe.pooling_mode,
        "probe_score": score[0],
        "probe_score_std": score[1],
        "probe_record_count": score[2],
        "probe_score_delta_vs_clean": score[0] - clean_score,
    }
    if future_metrics is None or clean_future_metrics is None:
        row.update(
            {
                "future_mae": float("nan"),
                "future_mse": float("nan"),
                "future_scaled_mae": float("nan"),
                "future_mae_ratio_vs_clean": float("nan"),
            }
        )
    else:
        row.update(future_metrics)
        clean_mae = clean_future_metrics["future_mae"]
        row["future_mae_ratio_vs_clean"] = (
            future_metrics["future_mae"] / clean_mae
            if clean_mae > 0
            else float("nan")
        )
    return row


@torch.inference_mode()
def run_pair(
    pipeline,
    *,
    triple: PairTriple,
    blends: tuple[float, ...],
    probe: ProbeArtifact,
    device: str,
    seq_len: int,
    patch_size: int,
    secondary_available: bool,
    split_seed: int,
    sampling_seed: int,
) -> list[dict[str, Any]]:
    layer = int(probe.layer)
    clean_residual, _ = _capture_forward(
        pipeline,
        layer=layer,
        forward=_embedding_forward(
            pipeline, triple.target, device=device, seq_len=seq_len
        ),
    )
    real_residual, _ = _capture_forward(
        pipeline,
        layer=layer,
        forward=_embedding_forward(
            pipeline, triple.real_source, device=device, seq_len=seq_len
        ),
    )
    null_residual, _ = _capture_forward(
        pipeline,
        layer=layer,
        forward=_embedding_forward(
            pipeline, triple.null_source, device=device, seq_len=seq_len
        ),
    )
    clean_score = _score_residual(clean_residual, window=triple.target, probe=probe)

    patched_scores: dict[tuple[float, str], tuple[float, float, int]] = {}
    for blend in blends:
        for condition, source_residual in (
            ("real_patch", real_residual),
            ("null_patch", null_residual),
        ):
            config = MomentInterchangeConfig(
                layer=layer,
                token_position=probe.token_position,
                source_residual=source_residual,
                blend=blend,
            )
            with apply_moment_interchange(pipeline, config):
                patched_residual, _ = _capture_forward(
                    pipeline,
                    layer=layer,
                    forward=_embedding_forward(
                        pipeline, triple.target, device=device, seq_len=seq_len
                    ),
                )
            patched_scores[(blend, condition)] = _score_residual(
                patched_residual,
                window=triple.target,
                probe=probe,
            )

    clean_future_metrics: dict[str, float] | None = None
    patched_future_metrics: dict[tuple[float, str], dict[str, float]] = {}
    if secondary_available:
        clean_reconstruction_residual, clean_output = _capture_forward(
            pipeline,
            layer=layer,
            forward=_short_forecast_forward(
                pipeline,
                triple.target,
                device=device,
                seq_len=seq_len,
                patch_size=patch_size,
            ),
        )
        clean_future_metrics = _future_error_metrics(
            clean_output, window=triple.target, patch_size=patch_size
        )
        real_reconstruction_residual, _ = _capture_forward(
            pipeline,
            layer=layer,
            forward=_short_forecast_forward(
                pipeline,
                triple.real_source,
                device=device,
                seq_len=seq_len,
                patch_size=patch_size,
            ),
        )
        null_reconstruction_residual, _ = _capture_forward(
            pipeline,
            layer=layer,
            forward=_short_forecast_forward(
                pipeline,
                triple.null_source,
                device=device,
                seq_len=seq_len,
                patch_size=patch_size,
            ),
        )
        # Keep the captured clean residual live in the protocol audit even
        # though only its unpatched output is needed for the clean error.
        if tuple(clean_reconstruction_residual.shape) != tuple(real_reconstruction_residual.shape):
            raise ValueError("Real reconstruction donor shape differs from its target.")
        if tuple(clean_reconstruction_residual.shape) != tuple(null_reconstruction_residual.shape):
            raise ValueError("Null reconstruction donor shape differs from its target.")

        for blend in blends:
            for condition, source_residual in (
                ("real_patch", real_reconstruction_residual),
                ("null_patch", null_reconstruction_residual),
            ):
                config = MomentInterchangeConfig(
                    layer=layer,
                    token_position=probe.token_position,
                    source_residual=source_residual,
                    blend=blend,
                )
                with apply_moment_interchange(pipeline, config):
                    _, output = _capture_forward(
                        pipeline,
                        layer=layer,
                        forward=_short_forecast_forward(
                            pipeline,
                            triple.target,
                            device=device,
                            seq_len=seq_len,
                            patch_size=patch_size,
                        ),
                    )
                patched_future_metrics[(blend, condition)] = _future_error_metrics(
                    output,
                    window=triple.target,
                    patch_size=patch_size,
                )

    rows: list[dict[str, Any]] = []
    for blend in blends:
        rows.append(
            _condition_row(
                triple=triple,
                blend=blend,
                condition="clean",
                source=None,
                score=clean_score,
                clean_score=clean_score[0],
                future_metrics=clean_future_metrics,
                clean_future_metrics=clean_future_metrics,
                split_seed=split_seed,
                sampling_seed=sampling_seed,
                probe=probe,
            )
        )
        for condition, source in (
            ("real_patch", triple.real_source),
            ("null_patch", triple.null_source),
        ):
            rows.append(
                _condition_row(
                    triple=triple,
                    blend=blend,
                    condition=condition,
                    source=source,
                    score=patched_scores[(blend, condition)],
                    clean_score=clean_score[0],
                    future_metrics=patched_future_metrics.get((blend, condition)),
                    clean_future_metrics=clean_future_metrics,
                    split_seed=split_seed,
                    sampling_seed=sampling_seed,
                    probe=probe,
                )
            )
    return rows


def validate_result_frame(
    frame: pd.DataFrame,
    *,
    num_pairs: int,
    blends: tuple[float, ...],
) -> None:
    if frame.empty:
        raise ValueError("MOMENT interchange produced no results.")
    key = ["split_seed", "sampling_seed", "pair_id", "blend", "condition"]
    if frame.duplicated(key).any():
        raise ValueError("MOMENT interchange results contain duplicate pair-condition rows.")
    expected_conditions = {"clean", "real_patch", "null_patch"}
    grouped = frame.groupby(["split_seed", "sampling_seed", "pair_id", "blend"])[
        "condition"
    ].agg(set)
    if not grouped.map(lambda values: values == expected_conditions).all():
        raise ValueError("Every pair/blend must contain one clean, real, and null row.")
    if frame["pair_id"].nunique() != num_pairs:
        raise ValueError(
            f"Expected {num_pairs} pair IDs, observed {frame['pair_id'].nunique()}."
        )
    if set(np.round(frame["blend"].astype(float), 12)) != set(np.round(blends, 12)):
        raise ValueError("Result blends disagree with the predeclared blend set.")
    if not frame.groupby("pair_id")["target_window_id"].nunique().eq(1).all():
        raise ValueError("A pair_id changes target window across conditions or blends.")
    source_stability = (
        frame[frame["condition"] != "clean"]
        .groupby(["pair_id", "condition"])["source_window_id"]
        .nunique()
    )
    if not source_stability.eq(1).all():
        raise ValueError("A pair_id changes its real/null donor across blends.")

    triples = frame[frame["condition"] != "clean"].pivot(
        index=["split_seed", "sampling_seed", "pair_id", "blend"],
        columns="condition",
        values=["target_window_id", "source_window_id", "target_series_id", "source_series_id"],
    )
    if (
        triples[("target_window_id", "real_patch")]
        != triples[("target_window_id", "null_patch")]
    ).any():
        raise ValueError("Real and null arms do not share exact target windows.")
    if (
        triples[("target_series_id", "real_patch")]
        == triples[("source_series_id", "real_patch")]
    ).any() or (
        triples[("target_series_id", "null_patch")]
        == triples[("source_series_id", "null_patch")]
    ).any():
        raise ValueError("A donor shares its target series.")
    if (
        triples[("source_series_id", "real_patch")]
        == triples[("source_series_id", "null_patch")]
    ).any():
        raise ValueError("Real and null donors share a series.")

    real_rows = (
        frame[frame["condition"] == "real_patch"]
        .sort_values(["pair_id", "blend"])
        .drop_duplicates("pair_id")
        .set_index("pair_id")
    )
    null_rows = (
        frame[frame["condition"] == "null_patch"]
        .sort_values(["pair_id", "blend"])
        .drop_duplicates("pair_id")
        .set_index("pair_id")
    )
    exact = pd.DataFrame(
        {
            "target_window_id": real_rows["target_window_id"],
            "real_source_window_id": real_rows["source_window_id"],
            "null_source_window_id": null_rows["source_window_id"],
        }
    )
    if exact.duplicated(
        ["target_window_id", "real_source_window_id", "null_source_window_id"]
    ).any():
        raise ValueError("Exact target/real/null triples are not unique.")


def pair_manifest_frame(triples: list[PairTriple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": triple.pair_id,
                "target_series_id": triple.target.series_id,
                "target_window_id": triple.target.window_id,
                "target_future_burstiness_label": _future_burstiness(triple.target),
                "real_source_series_id": triple.real_source.series_id,
                "real_source_window_id": triple.real_source.window_id,
                "real_source_future_burstiness_label": _future_burstiness(
                    triple.real_source
                ),
                "null_source_series_id": triple.null_source.series_id,
                "null_source_window_id": triple.null_source.window_id,
                "null_source_future_burstiness_label": _future_burstiness(
                    triple.null_source
                ),
                "num_variates": int(triple.target.context.shape[0]),
                "null_match_distance": triple.null_match_distance,
                "null_match_rank": triple.null_match_rank,
                "null_candidate_count": triple.null_candidate_count,
            }
            for triple in triples
        ]
    )


def summarize_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_summary = (
        frame.groupby(["blend", "condition"], sort=True)
        .agg(
            n_pairs=("pair_id", "nunique"),
            probe_score_mean=("probe_score", "mean"),
            probe_score_median=("probe_score", "median"),
            probe_delta_vs_clean_mean=("probe_score_delta_vs_clean", "mean"),
            future_mae_mean=("future_mae", "mean"),
            future_scaled_mae_mean=("future_scaled_mae", "mean"),
        )
        .reset_index()
    )

    pivot = frame.pivot(
        index=["split_seed", "sampling_seed", "pair_id", "blend"],
        columns="condition",
        values=["probe_score", "future_mae", "future_scaled_mae"],
    )
    rows: list[dict[str, Any]] = []
    for blend, group in pivot.groupby(level="blend", sort=True):
        probe_difference = (
            group[("probe_score", "real_patch")]
            - group[("probe_score", "null_patch")]
        )
        future_difference = (
            group[("future_mae", "real_patch")]
            - group[("future_mae", "null_patch")]
        )
        future_ratio = group[("future_mae", "real_patch")] / group[
            ("future_mae", "null_patch")
        ].clip(lower=1e-12)
        rows.append(
            {
                "blend": float(blend),
                "n_pairs": int(len(group)),
                "probe_real_minus_null_mean": float(probe_difference.mean()),
                "probe_real_minus_null_median": float(probe_difference.median()),
                "probe_real_over_null_win_fraction": float((probe_difference > 0).mean()),
                "probe_real_over_null_wins": int((probe_difference > 0).sum()),
                "probe_real_over_null_losses": int((probe_difference < 0).sum()),
                "probe_real_over_null_ties": int((probe_difference == 0).sum()),
                "future_mae_real_minus_null_mean": float(future_difference.mean()),
                "future_mae_real_minus_null_median": float(future_difference.median()),
                "future_mae_real_over_null_median": float(future_ratio.median()),
                "future_mae_real_better_fraction": float((future_difference < 0).mean()),
            }
        )
    return condition_summary, pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling_seed = args.split_seed if args.sampling_seed is None else args.sampling_seed
    blends = tuple(float(value) for value in args.blends)
    if len(set(blends)) != len(blends):
        raise ValueError(f"Blend values must be unique, received {blends}.")
    if any(value <= 0.0 or value > 1.0 for value in blends):
        raise ValueError(f"Blend values must lie in (0, 1], received {blends}.")

    selection = resolve_probe_selection(args)
    probe = ProbeArtifact.load(selection.path)
    summary = json.loads(args.activation_summary.read_text())
    validate_activation_summary(
        summary,
        split_seed=args.split_seed,
        model_id=args.model_id,
    )
    if int(probe.layer) not in {int(value) for value in summary.get("layers", [])}:
        raise ValueError(f"Probe layer {probe.layer} was not present in the activation dump.")
    if probe.token_position not in set(summary.get("token_positions", [])):
        raise ValueError(
            f"Probe token position {probe.token_position!r} was not in the activation dump."
        )
    if probe.pooling_mode not in set(summary.get("pooling_modes", [])):
        raise ValueError(
            f"Probe pooling mode {probe.pooling_mode!r} was not in the activation dump."
        )

    taxonomy = load_boom_taxonomy(local_path=args.snapshot_path)
    split_ids, split_manifest, split_overlap = reconstruct_and_validate_splits(
        taxonomy=taxonomy,
        summary=summary,
        split_seed=args.split_seed,
    )
    signature_raw, signature = validate_probe_provenance(
        probe,
        selection=selection,
        summary=summary,
        split_seed=args.split_seed,
        model_id=args.model_id,
        split_manifest=split_manifest,
    )

    seq_len = int(summary["context_length"])
    expected_patch_size = int(summary["patch_size"])
    max_windows_per_series = int(summary["max_windows_per_series"])
    snapshot_path = ensure_boom_snapshot(
        split_ids["test"],
        local_path=args.snapshot_path,
    )
    windows = build_boom_windows(
        series_ids=split_ids["test"],
        split="test",
        snapshot_path=snapshot_path,
        taxonomy=taxonomy,
        context_length=seq_len,
        patch_size=expected_patch_size,
        max_windows_per_series=max_windows_per_series,
        include_heldout_late=False,
    )
    logger.info(
        "Built %d held-out test windows for resplit seed %d",
        len(windows),
        args.split_seed,
    )
    triples, high_threshold, low_threshold = build_matched_pair_triples(
        windows,
        num_pairs=args.num_pairs,
        seed=sampling_seed,
        high_quantile=args.high_quantile,
        low_quantile=args.low_quantile,
        null_match_k=args.null_match_k,
    )

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    device = resolve_device(args.device)
    load_task = "embedding" if args.secondary_endpoint == "off" else "reconstruction"
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    pipeline = load_moment_with_fallback(
        args.model_id,
        device=device,
        weight_source="pretrained",
        task_name=load_task,
        seq_len=seq_len,
        dtype=dtype_map[args.dtype],
    )
    pipeline.eval()
    if pipeline.training:
        raise RuntimeError("MOMENT must remain in evaluation mode during interchange.")
    if moment_patch_size(pipeline) != expected_patch_size:
        raise ValueError("Loaded MOMENT patch size disagrees with activation provenance.")
    if int(probe.layer) >= moment_num_layers(pipeline):
        raise ValueError(f"Probe layer {probe.layer} is outside the loaded MOMENT encoder.")
    if probe.feature_mean is None or probe.feature_mean.numel() != moment_hidden_dim(pipeline):
        raise ValueError("Probe feature width disagrees with the loaded MOMENT hidden dimension.")

    has_short_forecast = callable(getattr(pipeline, "short_forecast", None))
    has_reconstruction_head = not isinstance(
        getattr(pipeline, "head", None), torch.nn.Identity
    )
    secondary_available = (
        args.secondary_endpoint != "off"
        and has_short_forecast
        and has_reconstruction_head
    )
    if args.secondary_endpoint == "required" and not secondary_available:
        raise RuntimeError(
            "The installed MOMENT API does not expose both short_forecast and "
            "a pretrained reconstruction head."
        )
    if args.secondary_endpoint == "auto" and not secondary_available:
        logger.warning(
            "Official MOMENT short-forecast reconstruction endpoint unavailable; "
            "writing probe-only results."
        )

    all_rows: list[dict[str, Any]] = []
    for triple in triples:
        all_rows.extend(
            run_pair(
                pipeline,
                triple=triple,
                blends=blends,
                probe=probe,
                device=device,
                seq_len=seq_len,
                patch_size=expected_patch_size,
                secondary_available=secondary_available,
                split_seed=args.split_seed,
                sampling_seed=sampling_seed,
            )
        )
        if (triple.pair_id + 1) % 5 == 0:
            logger.info(
                "Completed %d/%d exact matched triples",
                triple.pair_id + 1,
                len(triples),
            )
            pd.DataFrame(all_rows).to_csv(
                args.output_dir / "moment_interchange_results.csv",
                index=False,
            )

    result_frame = pd.DataFrame(all_rows)
    validate_result_frame(result_frame, num_pairs=args.num_pairs, blends=blends)
    result_path = args.output_dir / "moment_interchange_results.csv"
    result_frame.to_csv(result_path, index=False)

    pair_manifest = pair_manifest_frame(triples)
    pair_manifest_path = args.output_dir / "moment_interchange_pairs.csv"
    pair_manifest.to_csv(pair_manifest_path, index=False)
    condition_summary, paired_summary = summarize_results(result_frame)
    condition_summary.to_csv(
        args.output_dir / "moment_interchange_condition_summary.csv",
        index=False,
    )
    paired_summary.to_csv(
        args.output_dir / "moment_interchange_paired_summary.csv",
        index=False,
    )

    pair_exact = [
        (
            triple.target.window_id,
            triple.real_source.window_id,
            triple.null_source.window_id,
        )
        for triple in triples
    ]
    pair_triples_sha256 = hashlib.sha256(
        "\n".join("\t".join(values) for values in pair_exact).encode()
    ).hexdigest()
    meta = {
        "schema_version": 1,
        "protocol": "moment_base_matched_source_residual_interchange",
        "primary_endpoint": "probe_score_real_minus_null_within_target",
        "primary_endpoint_is_head_agnostic": True,
        "primary_endpoint_role": "same-site_probe_manipulation_check",
        "secondary_endpoint_requested": args.secondary_endpoint,
        "secondary_endpoint_available": secondary_available,
        "secondary_endpoint": (
            "official_moment_short_forecast_next_patch_error"
            if secondary_available
            else None
        ),
        "secondary_endpoint_omission_reason": (
            None
            if secondary_available
            else "disabled_by_user"
            if args.secondary_endpoint == "off"
            else "installed_api_missing_short_forecast_or_reconstruction_head"
        ),
        "secondary_final_token_semantics": (
            "masked_future_token_under_short_forecast"
            if secondary_available and probe.token_position == "final_context"
            else None
        ),
        "model_id": args.model_id,
        "weight_source": "pretrained",
        "model_task_loaded": load_task,
        "model_training": bool(pipeline.training),
        "device": device,
        "dtype": args.dtype,
        "split": "test",
        "split_seed": args.split_seed,
        "sampling_seed": sampling_seed,
        "seeds": [sampling_seed],
        "context_length": seq_len,
        "patch_size": expected_patch_size,
        "series_count": len(split_ids["test"]),
        "series_id_sha256": split_manifest["test"]["series_id_sha256"],
        "split_manifest": split_manifest,
        "split_overlap_counts": split_overlap,
        "n_windows": len(windows),
        "num_pairs_per_seed": args.num_pairs,
        "pair_triples_sha256": pair_triples_sha256,
        "unique_exact_pair_triples": True,
        "same_pair_triples_across_blends": True,
        "common_targets_across_conditions": True,
        "cross_series_sources": True,
        "exact_channel_count_matching": True,
        "implicit_source_broadcasting": False,
        "deterministic_eval_forward": True,
        "blends": list(blends),
        "high_quantile": args.high_quantile,
        "low_quantile": args.low_quantile,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "null_pool_rule": "future_burstiness_below_high_threshold",
        "null_match_k": args.null_match_k,
        "null_match_numeric_features": [
            "last_patch_mean",
            "last_patch_std",
            "last_patch_zero_fraction",
            "last_patch_abs_mean",
            "last_patch_abs_max",
            "last_patch_variate_mean_std",
            "current_sparsity",
            "current_burstiness",
        ],
        "null_match_categorical_penalties": list(MATCH_CATEGORICAL_FIELDS),
        "probe_selection_rule": selection.rule,
        "probe_selection_metric": selection.validation_metric,
        "probe_selection_score": selection.validation_score,
        "probe_path": str(selection.path),
        "probe_sha256": sha256_file(selection.path),
        "probe_label": probe.label_spec.name,
        "probe_seed": probe.seed,
        "probe_model_id": probe.model_id,
        "probe_weight_source": probe.weight_source,
        "probe_layer": probe.layer,
        "probe_token_position": probe.token_position,
        "probe_pooling_mode": probe.pooling_mode,
        "probe_activation_source_signature": signature_raw,
        "probe_activation_source": signature,
        "activation_summary_path": str(args.activation_summary.resolve()),
        "activation_summary_sha256": sha256_file(args.activation_summary),
        "results_path": str(result_path),
        "results_sha256": sha256_file(result_path),
        "pair_manifest_path": str(pair_manifest_path),
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
    }
    meta_path = args.output_dir / "moment_interchange_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("Done: %d rows -> %s", len(result_frame), result_path)


if __name__ == "__main__":
    main()
