from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Toto activation traces for BOOM research windows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-windows-per-series", type=int, default=16)
    parser.add_argument("--max-series-per-split", type=int, default=0)
    parser.add_argument("--series-start", type=int, default=0)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--include-heldout-late", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=None,
        help="Local BOOM dataset directory; skips HuggingFace download.",
    )
    parser.add_argument(
        "--token-positions",
        nargs="+",
        choices=("all_context", "final_context", "first_decode"),
        default=["all_context", "final_context", "first_decode"],
    )
    parser.add_argument(
        "--pooling-modes",
        nargs="+",
        choices=("per_variate", "series_mean"),
        default=["per_variate", "series_mean"],
    )
    parser.add_argument(
        "--weight-source",
        choices=("pretrained", "layer_permuted_pretrained", "random_init", "checkpoint"),
        default="pretrained",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--randomize-scope", choices=("full", "selected_layers", "head_only"), default="full")
    parser.add_argument("--randomize-layers", type=int, nargs="*", default=[])
    parser.add_argument(
        "--layer-permutation-seed",
        type=int,
        default=None,
        help=(
            "Seed for layer_permuted_pretrained. Defaults to --seed so each split resample "
            "has a fully recorded, deterministic block ordering."
        ),
    )
    return parser.parse_args()


def seed_process_randomness(seed: int) -> None:
    """Seed the process before any randomized-backbone initialization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_process_randomness(args.seed)
    from toto_interp import (
        TraceConfig,
        WindowDataset,
        extract_activations,
        get_toto_weight_provenance,
        load_toto_with_fallback,
    )
    from toto_interp.boom import (
        build_boom_windows,
        ensure_boom_snapshot,
        load_boom_taxonomy,
        split_boom_series_ids,
    )
    from toto_interp.loader import resolve_device

    device = resolve_device(args.device)
    if args.weight_source != "layer_permuted_pretrained" and args.layer_permutation_seed is not None:
        raise ValueError("--layer-permutation-seed is only valid with --weight-source layer_permuted_pretrained.")
    if args.weight_source == "layer_permuted_pretrained" and args.randomize_layers:
        raise ValueError("Layer permutation preserves pretrained blocks; do not combine it with --randomize-layers.")
    layer_permutation_seed = args.seed if args.weight_source == "layer_permuted_pretrained" else None
    if args.layer_permutation_seed is not None:
        layer_permutation_seed = args.layer_permutation_seed
    reported_randomize_scope = args.randomize_scope if args.weight_source == "random_init" else None
    reported_randomize_layers = args.randomize_layers if args.weight_source == "random_init" else []
    model = load_toto_with_fallback(
        args.model_id,
        map_location="cpu",
        device=device,
        weight_source=args.weight_source,
        checkpoint_path=args.checkpoint_path,
        randomize_scope=args.randomize_scope,
        randomize_layers=tuple(args.randomize_layers),
        layer_permutation_seed=layer_permutation_seed,
    )
    if model.training:
        raise RuntimeError("Toto must be in evaluation mode during activation extraction.")
    weight_provenance = get_toto_weight_provenance(model)
    if args.compile and hasattr(model, "compile"):
        model.compile()
    backbone = model.model
    patch_size = int(backbone.patch_embed.patch_size)

    if args.context_length % patch_size != 0:
        raise ValueError(f"context-length must be divisible by patch size ({patch_size}).")

    taxonomy = load_boom_taxonomy(local_path=args.snapshot_path)
    split_ids = split_boom_series_ids(taxonomy, seed=args.seed)
    if args.max_series_per_split > 0:
        split_ids = {
            split_name: series_ids[args.series_start : args.series_start + args.max_series_per_split]
            for split_name, series_ids in split_ids.items()
        }
    elif args.series_start > 0:
        split_ids = {split_name: series_ids[args.series_start :] for split_name, series_ids in split_ids.items()}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    split_overlap_counts = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(split_overlap_counts.values()):
        raise RuntimeError(f"BOOM series splits overlap: {split_overlap_counts}")
    series_hashes = {
        name: hashlib.sha256("\n".join(ids).encode()).hexdigest()
        for name, ids in split_ids.items()
    }

    snapshot_path = ensure_boom_snapshot(
        [series_id for series_ids in split_ids.values() for series_id in series_ids],
        local_path=args.snapshot_path,
    )

    trace_config = TraceConfig(
        token_positions=tuple(args.token_positions),
        pooling_modes=tuple(args.pooling_modes),
        use_kv_cache=not args.disable_kv_cache,
    )
    summary: dict[str, object] = {
        "seed": args.seed,
        "model_id": args.model_id,
        "weight_source": args.weight_source,
        "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
        "randomize_scope": reported_randomize_scope,
        "randomize_layers": reported_randomize_layers,
        "layer_permutation_seed": layer_permutation_seed,
        "weight_provenance": weight_provenance,
        "patch_size": patch_size,
        "context_length": args.context_length,
        "max_windows_per_series": args.max_windows_per_series,
        "max_series_per_split": args.max_series_per_split,
        "series_start": args.series_start,
        "snapshot_path": None if args.snapshot_path is None else str(args.snapshot_path),
        "device": device,
        "model_training": bool(model.training),
        "split_overlap_counts": split_overlap_counts,
        "series_id_sha256_by_split": series_hashes,
        "trace_config": {
            "layers": trace_config.layers,
            "token_positions": trace_config.token_positions,
            "pooling_modes": trace_config.pooling_modes,
            "use_kv_cache": trace_config.use_kv_cache,
        },
        "splits": {},
    }

    for split_name, series_ids in split_ids.items():
        if split_name not in args.splits:
            continue
        windows = build_boom_windows(
            series_ids=series_ids,
            split=split_name,
            snapshot_path=snapshot_path,
            taxonomy=taxonomy,
            context_length=args.context_length,
            patch_size=patch_size,
            max_windows_per_series=args.max_windows_per_series,
            include_heldout_late=args.include_heldout_late,
        )
        source_metadata = {
            "model_id": args.model_id,
            "weight_source": args.weight_source,
            "checkpoint_path": None if args.checkpoint_path is None else str(args.checkpoint_path),
            "randomize_scope": reported_randomize_scope,
            "randomize_layers": reported_randomize_layers,
            "layer_permutation_seed": layer_permutation_seed,
            "weight_provenance": weight_provenance,
            "seed": args.seed,
            "context_length": args.context_length,
            "patch_size": patch_size,
            "max_windows_per_series": args.max_windows_per_series,
            "max_series_per_split": args.max_series_per_split,
            "series_start": args.series_start,
            "snapshot_path": None if args.snapshot_path is None else str(args.snapshot_path),
            "model_training": bool(model.training),
            "split_overlap_counts": split_overlap_counts,
            "series_id_sha256_by_split": series_hashes,
        }
        batch = extract_activations(model, windows, trace_config)
        batch.source_metadata.update(source_metadata)
        batch_path = args.output_dir / f"{split_name}_activations.pt"
        batch.save(batch_path)
        window_dataset = WindowDataset.from_windows(windows, source_metadata=source_metadata)
        window_path = args.output_dir / f"{split_name}_windows.pt"
        window_dataset.save(window_path)
        summary["splits"][split_name] = {
            "series_count": len(series_ids),
            "series_id_sha256": series_hashes[split_name],
            "window_count": len(windows),
            "activation_count": len(batch),
            "path": str(batch_path),
            "window_dataset_path": str(window_path),
        }

    with open(args.output_dir / "activation_dump_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
