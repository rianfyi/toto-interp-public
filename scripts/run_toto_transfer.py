from __future__ import annotations

import argparse
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

from toto_interp import ProbeArtifact, TraceConfig, extract_activations, load_toto_with_fallback, score_probe
from toto_interp.fev_tasks import get_fev_task, list_fev_tasks
from toto_interp.loader import resolve_device
from toto_interp.lsf import default_lsf_data_path, ensure_lsf_datasets, required_archives_for_lsf_datasets
from toto_interp.transfer import build_fev_windows, build_lsf_windows


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Run zero-shot BOOM probe transfer on FEV and LSF datasets.")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--weight-source", choices=("pretrained", "random_init", "checkpoint"), default="pretrained")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--randomize-scope", choices=("full", "selected_layers", "head_only"), default="full")
    parser.add_argument("--randomize-layers", type=int, nargs="*", default=[])
    parser.add_argument("--probe-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--probe-dir", type=Path, default=None)
    parser.add_argument("--dataset", choices=("fev", "lsf", "both"), default="both")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-series", type=int, default=8)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--include-heldout-late", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--fev-tasks", nargs="*", default=[])
    parser.add_argument("--fev-safe-only", action="store_true")
    parser.add_argument("--list-fev-tasks", action="store_true")
    parser.add_argument("--fev-configs", nargs="*", default=[])
    parser.add_argument("--fev-target-fields", nargs="*", default=None)
    parser.add_argument("--fev-ev-fields", nargs="*", default=[])
    parser.add_argument("--lsf-datasets", nargs="*", default=["ETTh1"])
    parser.add_argument("--lsf-path", type=Path, default=None)
    parser.add_argument("--download-lsf", action="store_true")
    parser.add_argument(
        "--require-eval-provenance",
        action="store_true",
        help="Reject probes whose activation signature does not record eval-mode extraction.",
    )
    parser.add_argument(
        "--validation-select-one-per-label",
        action="store_true",
        help=(
            "Select one continuous probe view per label using BOOM validation R2 only. "
            "Ties are resolved deterministically by layer, token position, and pooling mode."
        ),
    )
    return parser.parse_args(), parser


# The six continuous dynamic labels eligible for zero-shot probe transfer.
_ALLOWED_TRANSFER_LABELS = frozenset({
    "current_sparsity",
    "future_sparsity",
    "current_burstiness",
    "future_burstiness",
    "shift_risk",
    "coordination",
})


def resolve_probe_paths(args: argparse.Namespace) -> list[Path]:
    if args.probe_paths:
        if args.validation_select_one_per_label:
            raise ValueError(
                "--validation-select-one-per-label requires --probe-dir so the BOOM "
                "validation table can be audited."
            )
        return list(args.probe_paths)
    if args.probe_dir is None:
        raise ValueError("Provide either --probe-paths or --probe-dir.")

    if args.validation_select_one_per_label:
        results_path = args.probe_dir / "probe_results.csv"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Validation selection requires probe results at {results_path}."
            )
        frame = pd.read_csv(results_path)
        required = {
            "label",
            "task_type",
            "method",
            "layer",
            "token_position",
            "pooling_mode",
            "val_r2",
            "artifact_path",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{results_path} is missing validation-selection columns: {sorted(missing)}"
            )
        frame = frame[
            (frame["task_type"] == "continuous")
            & (frame["method"] == "linear_probe")
            & frame["label"].isin(_ALLOWED_TRANSFER_LABELS)
        ].copy()
        frame["val_r2"] = pd.to_numeric(frame["val_r2"], errors="coerce")
        frame = frame[np.isfinite(frame["val_r2"])].sort_values(
            ["label", "val_r2", "layer", "token_position", "pooling_mode"],
            ascending=[True, False, True, True, True],
        )
        selected = frame.groupby("label", sort=True, as_index=False).head(1)
        if selected.empty:
            raise ValueError(f"{results_path} has no validation-scored continuous probes.")
        paths: list[Path] = []
        for artifact in selected["artifact_path"]:
            recorded = Path(str(artifact))
            candidates = (
                recorded,
                args.probe_dir / "artifacts" / recorded.name,
            )
            resolved = next((path for path in candidates if path.exists()), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"Validation-selected probe artifact does not exist: {recorded}"
                )
            paths.append(resolved)
        return paths

    localization_path = args.probe_dir / "probe_localization_summary.csv"
    if localization_path.exists():
        df = pd.read_csv(localization_path)
        df = df[(df["task_type"] == "continuous") & (df.get("method", "linear_probe") == "linear_probe")]
        if "label" in df.columns:
            df = df[df["label"].isin(_ALLOWED_TRANSFER_LABELS)]
        return [Path(path) for path in df["artifact_path"].tolist()]

    artifact_paths = sorted((args.probe_dir / "artifacts").glob("*.pt"))
    filtered_paths: list[Path] = []
    for path in artifact_paths:
        probe = ProbeArtifact.load(path)
        if probe.method == "linear_probe":
            filtered_paths.append(path)
    return filtered_paths


def finite_windows(windows: list) -> tuple[list, list[str]]:
    eligible: list = []
    excluded_window_ids: list[str] = []
    for window in windows:
        if torch.isfinite(window.context).all() and torch.isfinite(window.next_patch).all():
            eligible.append(window)
        else:
            excluded_window_ids.append(str(window.window_id))
    return eligible, excluded_window_ids


def build_windows_for_dataset(
    args: argparse.Namespace,
    *,
    benchmark_name: str,
    patch_size: int,
) -> dict[str, list]:
    datasets_to_windows: dict[str, list] = {}
    if benchmark_name == "fev":
        task_names = list(args.fev_tasks)
        if args.fev_safe_only:
            task_names.extend(task.dataset_config for task in list_fev_tasks(safe_only=True))
        for task_name in sorted(set(task_names)):
            task_spec = get_fev_task(task_name)
            if task_spec is None:
                raise ValueError(f"Unknown FEV task: {task_name}")
            datasets_to_windows[task_name] = build_fev_windows(
                config_name=task_spec.dataset_config,
                task_name=task_name,
                context_length=args.context_length,
                patch_size=patch_size,
                target_fields=task_spec.target_fields,
                ev_fields=task_spec.exogenous_fields,
                max_series=args.max_series,
                max_windows_per_series=args.max_windows_per_series,
                include_heldout_late=args.include_heldout_late,
            )
        for config_name in args.fev_configs:
            datasets_to_windows[config_name] = build_fev_windows(
                config_name=config_name,
                context_length=args.context_length,
                patch_size=patch_size,
                target_fields=args.fev_target_fields,
                ev_fields=args.fev_ev_fields,
                max_series=args.max_series,
                max_windows_per_series=args.max_windows_per_series,
                include_heldout_late=args.include_heldout_late,
            )
    elif benchmark_name == "lsf":
        if args.lsf_path is None:
            raise ValueError("--lsf-path is required for LSF transfer evaluation.")
        for dataset_name in args.lsf_datasets:
            datasets_to_windows[dataset_name] = build_lsf_windows(
                dataset_name=dataset_name,
                context_length=args.context_length,
                patch_size=patch_size,
                lsf_path=args.lsf_path,
                max_series=args.max_series,
                max_windows_per_series=args.max_windows_per_series,
                include_heldout_late=args.include_heldout_late,
            )
    else:
        raise ValueError(f"Unsupported benchmark name: {benchmark_name}")
    return datasets_to_windows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, parser = parse_args()
    if args.list_fev_tasks:
        for task in list_fev_tasks(safe_only=args.fev_safe_only):
            print(
                f"{task.dataset_config}\thorizon={task.horizon}\ttargets={','.join(task.target_fields)}"
                f"\tsafe={task.safe_for_paper}"
            )
        return

    if args.output_dir is None:
        parser.error("--output-dir is required unless --list-fev-tasks is used.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("lsf", "both"):
        resolved_lsf_path = args.lsf_path or default_lsf_data_path(ROOT)
        archive_keys = required_archives_for_lsf_datasets(args.lsf_datasets)
        ensure_lsf_datasets(
            resolved_lsf_path,
            archive_keys=archive_keys,
            download=args.download_lsf,
        )
        args.lsf_path = resolved_lsf_path

    probe_entries = [(path, ProbeArtifact.load(path)) for path in resolve_probe_paths(args)]
    probe_entries = [
        (path, probe)
        for path, probe in probe_entries
        if probe.label_spec.task_type == "continuous" and probe.method == "linear_probe"
    ]
    if not probe_entries:
        raise ValueError("No continuous probe artifacts were available for transfer evaluation.")
    if args.require_eval_provenance:
        for path, probe in probe_entries:
            signature_raw = probe.artifact_metadata.get("activation_source_signature")
            try:
                signature = json.loads(signature_raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Probe {path} has no parseable activation provenance") from exc
            if signature.get("model_training") is not False:
                raise ValueError(f"Probe {path} was not fit from eval-mode activations")
    probes = [probe for _, probe in probe_entries]

    device = resolve_device(args.device)
    model = load_toto_with_fallback(
        args.model_id,
        map_location="cpu",
        device=device,
        weight_source=args.weight_source,
        checkpoint_path=args.checkpoint_path,
        randomize_scope=args.randomize_scope,
        randomize_layers=tuple(args.randomize_layers),
    )
    if model.training:
        raise RuntimeError("Toto must be in evaluation mode during transfer scoring.")
    if args.compile and hasattr(model, "compile"):
        model.compile()
    patch_size = int(model.model.patch_embed.patch_size)
    if args.context_length % patch_size != 0:
        raise ValueError(f"context-length must be divisible by patch size ({patch_size}).")

    trace_config = TraceConfig(
        layers=tuple(sorted({probe.layer for probe in probes if probe.layer >= 0})),
        token_positions=tuple(sorted({probe.token_position for probe in probes})),
        pooling_modes=tuple(sorted({probe.pooling_mode for probe in probes})),
        capture_patch_embedding=any(probe.layer == -1 for probe in probes),
        use_kv_cache=not args.disable_kv_cache,
    )

    benchmarks: list[str]
    if args.dataset == "both":
        benchmarks = ["fev", "lsf"]
    else:
        benchmarks = [args.dataset]

    rows: list[dict[str, object]] = []
    dataset_meta: dict[str, dict[str, object]] = {}
    for benchmark_name in benchmarks:
        for dataset_name, windows in build_windows_for_dataset(args, benchmark_name=benchmark_name, patch_size=patch_size).items():
            candidate_count = len(windows)
            windows, excluded_window_ids = finite_windows(windows)
            dataset_meta[f"{benchmark_name}:{dataset_name}"] = {
                "candidate_window_count": candidate_count,
                "finite_window_count": len(windows),
                "excluded_nonfinite_window_count": len(excluded_window_ids),
                "excluded_nonfinite_window_ids": excluded_window_ids,
                "series_count": len({window.series_id for window in windows}),
                "eligibility_rule": "finite_context_and_next_patch_complete_case",
            }
            if not windows:
                continue
            activation_batch = extract_activations(model, windows, trace_config)
            if not torch.isfinite(activation_batch.activations).all():
                raise ValueError(
                    f"Non-finite residual activations for {benchmark_name}:{dataset_name} "
                    "after complete-case filtering."
                )
            for probe_path, probe in probe_entries:
                try:
                    metrics = score_probe(activation_batch, probe, prefix="transfer")
                except ValueError as e:
                    logging.warning(
                        "Skipping probe %r on %s:%s — %s",
                        probe.label_spec.name,
                        benchmark_name,
                        dataset_name,
                        e,
                    )
                    continue
                row = {
                    "benchmark": benchmark_name,
                    "dataset_name": dataset_name,
                    "probe_label": probe.label_spec.name,
                    "probe_path": str(probe_path),
                    "probe_seed": probe.seed,
                    "probe_weight_source": probe.weight_source,
                    "layer": probe.layer,
                    "token_position": probe.token_position,
                    "pooling_mode": probe.pooling_mode,
                }
                row.update(metrics)
                rows.append(row)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["benchmark", "dataset_name", "probe_label", "layer", "token_position"])
    results_df.to_csv(args.output_dir / "transfer_probe_metrics.csv", index=False)

    aggregations: dict[str, tuple[str, str]] = {"dataset_count": ("dataset_name", "nunique")}
    if not results_df.empty:
        for column in (
            "transfer_r2",
            "baseline_transfer_r2",
            "transfer_rmse",
            "baseline_transfer_rmse",
            "transfer_mae",
            "baseline_transfer_mae",
            "transfer_count",
        ):
            if column in results_df.columns:
                aggregations[column] = (column, "mean")
        summary_df = results_df.groupby(["benchmark", "probe_label"], dropna=False).agg(**aggregations).reset_index()
    else:
        summary_df = pd.DataFrame(columns=["benchmark", "probe_label", "dataset_count"])
    summary_df.to_csv(args.output_dir / "transfer_summary.csv", index=False)

    with open(args.output_dir / "transfer_meta.json", "w") as handle:
        json.dump(
            {
                "model_id": args.model_id,
                "probe_paths": [str(path) for path, _ in probe_entries],
                "dataset_mode": args.dataset,
                "device": device,
                "weight_source": args.weight_source,
                "model_training": bool(model.training),
                "require_eval_provenance": args.require_eval_provenance,
                "probe_selection_rule": (
                    "max_boom_validation_r2_per_label_with_deterministic_tie_break"
                    if args.validation_select_one_per_label
                    else "all_resolved_probe_artifacts"
                ),
                "external_window_eligibility_rule": (
                    "finite_context_and_next_patch_complete_case"
                ),
                "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
                "randomize_scope": args.randomize_scope,
                "randomize_layers": args.randomize_layers,
                "trace_config": {
                    "layers": trace_config.layers,
                    "token_positions": trace_config.token_positions,
                    "pooling_modes": trace_config.pooling_modes,
                    "use_kv_cache": trace_config.use_kv_cache,
                },
                "datasets": dataset_meta,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
